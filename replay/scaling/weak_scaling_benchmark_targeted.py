#!/usr/bin/env python3
import os
import time
import math
import csv
import subprocess
from statistics import mean, pstdev

# Configure FIRST_PERCENTAGE and STEP; we force-add 100% at the end if missing.
FIRST_PERCENTAGE = 20
PERCENTAGE_STEP = 20
INCLUDE_100 = True

# Target GPU-phase time and tolerance (fraction)
TARGET_GPU_SECONDS = 200.0
TOL_FRAC = 0.02          # 5% band

# Replicates to gather at the chosen point per percentage
REPLICATES_PER_PERCENT = 1

# Search parameters for finding -M that yields ~TARGET_GPU_SECONDS
M_INITIAL = 1000        # starting guess for the very first percentage
M_MIN = 1_000
M_MAX = 1_300_000
GROWTH_FACTOR = 1.1      # multiplicative growth when below lower bound, turned WAY down because we have good guesses
MAX_TRIALS_PER_PERCENT = 20
SLEEP_BETWEEN_RUNS = 1.0 # seconds; small pause to reduce jitter

GOOD_GUESSES = {20:32000, 40:56000, 60:70000, 80:80000, 100:90000}

# Base command
REPLAY_CMD = [
    "./replay",
    "networks/dense_2500.txt",
    "--step-size", "3600",
    "--iterations", "100",
    "--M", "1000",             # placeholder; will be replaced
    "--initial-infected", "0.5",
    "--infect-prob", "1.0",
    "--exposed-duration", "3600",
    "--infectious-duration", "3600",
    "--resistant-duration", "0",
    "--start-time", "946713600",
    "--static-network-duration", "3600",
]

# GPU / I/O behavior
CUDA_VISIBLE_DEVICES = "0"
USE_STDBUF = False
STDBUF_PREFIX = ["stdbuf", "-oL", "-eL"]

# Marker seen in stdout when GPU phase begins
START_MARKER_SUBSTR = "Starting temporal Monte Carlo simulation for"

# Output CSVs
CSV_RUNS = "weak_scaling_runs.csv"
CSV_SUMMARY = "weak_scaling_summary.csv"

# ========= End of config =========


def start_mps_daemon():
    print("Starting NVIDIA MPS daemon (sudo nvidia-cuda-mps-control -d)...")
    try:
        subprocess.run(["sudo", "nvidia-cuda-mps-control", "-d"], check=True)
    except subprocess.CalledProcessError as e:
        print("Warning: Failed to start MPS daemon. Make sure MPS is enabled.")
        print(f"Error: {e}")


def build_cmd_with_M(M):
    cmd = list(REPLAY_CMD)
    # Replace existing --M value if present; otherwise append.
    try:
        idx = cmd.index("--M")
        if idx + 1 < len(cmd):
            cmd[idx + 1] = str(M)
        else:
            cmd += [str(M)]
    except ValueError:
        cmd += ["--M", str(M)]
    return (STDBUF_PREFIX + cmd) if USE_STDBUF else cmd


def run_replay_once(percentage, M):
    """
    Run replay once with given CUDA_MPS_ACTIVE_THREAD_PERCENTAGE and -M.
    Returns dict with times (s) or None on failure.
    """
    print(f"\n=== Run: MPS={percentage}%  -M={M} ===")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(percentage)

    cmd = build_cmd_with_M(M)

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    t1 = None  # when we first see the GPU start marker
    try:
        for line in proc.stdout:
            if not line:
                continue
            line = line.rstrip("\n")
            print(line)
            if t1 is None and START_MARKER_SUBSTR in line:
                t1 = time.perf_counter()
    except Exception as e:
        print(f"Error while reading output: {e}")

    rc = proc.wait()
    t2 = time.perf_counter()

    total_time = t2 - t0
    if rc != 0:
        print(f"Return code {rc}. Elapsed {total_time:.3f} s.")
        return None

    if t1 is None:
        print(f"Warning: did not see start marker '{START_MARKER_SUBSTR}'.")
        return {
            "network_init": None,
            "gpu_time": None,
            "total_time": total_time,
            "return_code": rc,
        }

    network_init = t1 - t0
    gpu_time = t2 - t1

    print(f"Network init: {network_init:.3f} s | GPU phase: {gpu_time:.3f} s | Total: {total_time:.3f} s")
    return {
        "network_init": network_init,
        "gpu_time": gpu_time,
        "total_time": total_time,
        "return_code": rc,
    }


def within_band(gpu_time, target, tol_frac):
    if gpu_time is None:
        return False
    lo = target * (1.0 - tol_frac)
    hi = target * (1.0 + tol_frac)
    return (gpu_time >= lo) and (gpu_time <= hi)


def clamp_int(x, lo, hi):
    return max(lo, min(hi, int(x)))


def search_M_for_percentage(percentage, start_M, log_rows):
    """
    Searches for an M such that GPU time ~= TARGET_GPU_SECONDS (± TOL_FRAC).
    Strategy:
      - Evaluate start_M
      - If too fast (below lower bound): multiply by GROWTH_FACTOR until bracket or within range
      - If too slow (above upper bound): shrink by 1/GROWTH_FACTOR until bracket or within range
      - If bracketed: bisection on M
    Records every run into log_rows with phase='search'.
    Returns: (chosen_M, trials_made)
    """
    lower = TARGET_GPU_SECONDS * (1.0 - TOL_FRAC)
    upper = TARGET_GPU_SECONDS * (1.0 + TOL_FRAC)

    seen = set()
    trials = []

    M_low, t_low = None, None   # t_low < lower (too fast)
    M_high, t_high = None, None # t_high > upper (too slow)

    M = clamp_int(start_M, M_MIN, M_MAX)
    for trial_idx in range(1, MAX_TRIALS_PER_PERCENT + 1):
        # Avoid repeated M
        while M in seen:
            M = min(M_MAX, M + 1)
        seen.add(M)

        res = run_replay_once(percentage, M)
        time.sleep(SLEEP_BETWEEN_RUNS)

        # Log the attempt
        log_rows.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "percentage": percentage,
            "M": M,
            "phase": "search",
            "trial": trial_idx,
            "network_init_s": "" if res is None or res["network_init"] is None else f"{res['network_init']:.6f}",
            "gpu_time_s": "" if res is None or res["gpu_time"] is None else f"{res['gpu_time']:.6f}",
            "total_time_s": "" if res is None else f"{res['total_time']:.6f}",
            "return_code": "" if res is None else res["return_code"],
            "note": "",
        })

        trials.append((M, res))
        if res is None or res["gpu_time"] is None:
            # Could not measure GPU phase; try growing a bit to avoid extremely fast corner cases
            M = clamp_int(M * GROWTH_FACTOR, M_MIN, M_MAX)
            continue

        t = res["gpu_time"]
        if within_band(t, TARGET_GPU_SECONDS, TOL_FRAC):
            return M, trials

        if t < lower:
            # Too fast: not enough work, increase M
            M_low, t_low = M, t
            if M_high is None:
                M = clamp_int(math.ceil(M * GROWTH_FACTOR), M_MIN, M_MAX)
            else:
                # Bisect upward
                M = clamp_int((M + M_high) // 2, M_MIN, M_MAX)
        else:  # t > upper
            # Too slow: too much work, decrease M
            M_high, t_high = M, t
            if M_low is None:
                M = clamp_int(max(M_MIN, M / GROWTH_FACTOR), M_MIN, M_MAX)
            else:
                # Bisect downward
                M = clamp_int((M_low + M) // 2, M_MIN, M_MAX)

    # If we exit without hitting the band, pick the closest by absolute error
    def err(res):
        return float("inf") if (res is None or res["gpu_time"] is None) else abs(res["gpu_time"] - TARGET_GPU_SECONDS)

    best = min(trials, key=lambda kv: err(kv[1]))
    return best[0], trials


def write_csv(path, rows, header):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if new_file:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    # Build percentage list
    percentages = list(range(FIRST_PERCENTAGE, 100, PERCENTAGE_STEP))
    if INCLUDE_100 and 100 not in percentages:
        percentages.append(100)

    start_mps_daemon()

    # Optional warm-up
    print("Warm-up run at 100% (not recorded in CSVs)...")
    _ = run_replay_once(100, M_INITIAL)
    time.sleep(SLEEP_BETWEEN_RUNS)

    # Prepare CSV headers
    runs_header = [
        "timestamp", "percentage", "M", "phase", "trial",
        "network_init_s", "gpu_time_s", "total_time_s", "return_code", "note"
    ]
    summary_header = [
        "percentage", "chosen_M", "replicates",
        "mean_gpu_time_s", "std_gpu_time_s",
        "mean_total_time_s"
    ]

    all_runs_buffer = []  # accumulate and flush periodically
    summary_rows = []

    last_chosen_M = M_INITIAL

    for p in percentages:
        print(f"\n===== Searching for -M at MPS {p}% to hit ~{TARGET_GPU_SECONDS}s =====")

        # jump to good guesse if we already know it for this percentage - added manually from past observations
        if p in GOOD_GUESSES:
            last_chosen_M = GOOD_GUESSES[p]
        
        chosen_M, search_trials = search_M_for_percentage(p, last_chosen_M, all_runs_buffer)
        print(f"Chosen -M for {p}%: {chosen_M}")

        # Replicates at chosen point
        gpu_times = []
        total_times = []
        for r in range(1, REPLICATES_PER_PERCENT + 1):
            res = run_replay_once(p, chosen_M)
            time.sleep(SLEEP_BETWEEN_RUNS)
            all_runs_buffer.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "percentage": p,
                "M": chosen_M,
                "phase": "replicate",
                "trial": r,
                "network_init_s": "" if res is None or res["network_init"] is None else f"{res['network_init']:.6f}",
                "gpu_time_s": "" if res is None or res["gpu_time"] is None else f"{res['gpu_time']:.6f}",
                "total_time_s": "" if res is None else f"{res['total_time']:.6f}",
                "return_code": "" if res is None else res["return_code"],
                "note": "",
            })
            if res is not None and res["gpu_time"] is not None:
                gpu_times.append(res["gpu_time"])
                total_times.append(res["total_time"])

        # Flush per-percentage to survive interruptions
        write_csv(CSV_RUNS, all_runs_buffer, runs_header)
        all_runs_buffer.clear()

        # Summaries (use population stdev; fall back to 0 with 0/1 samples)
        mean_gpu = mean(gpu_times) if gpu_times else ""
        std_gpu = pstdev(gpu_times) if len(gpu_times) > 1 else (0.0 if len(gpu_times) == 1 else "")
        mean_total = mean(total_times) if total_times else ""

        summary_rows.append({
            "percentage": p,
            "chosen_M": chosen_M,
            "replicates": len(gpu_times),
            "mean_gpu_time_s": "" if mean_gpu == "" else f"{mean_gpu:.6f}",
            "std_gpu_time_s": "" if std_gpu == "" else f"{std_gpu:.6f}",
            "mean_total_time_s": "" if mean_total == "" else f"{mean_total:.6f}",
        })
        write_csv(CSV_SUMMARY, summary_rows[-1:], summary_header)

        # Carry chosen M forward
        last_chosen_M = chosen_M

    print("\nDone.")
    print(f"Runs CSV:     {CSV_RUNS}")
    print(f"Summary CSV:  {CSV_SUMMARY}")


if __name__ == "__main__":
    main()

