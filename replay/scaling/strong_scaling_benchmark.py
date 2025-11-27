#!/usr/bin/env python3
import subprocess
import time
import os

# Percentages to test for strong scaling: 10, 20, ..., 100
PERCENTAGES = list(reversed([1,2,4,8,16,32,64,100]))#list(range(10, 101, 10))

# Number of runs per percentage
RUNS_PER_PERCENTAGE = 1

# Base command (without env vars)
REPLAY_CMD = [
    "./replay",
    "networks/dense_2500.txt",
    "--step-size", "1200", # used to be 3600
    "--iterations", "300", # used to be 100
    "--M", "100000", # used to be 10000
    "--initial-infected", "0.5",
    "--infect-prob", "1.0",
    "--exposed-duration", "3600",
    "--infectious-duration", "3600",
    "--resistant-duration", "0",
    "--start-time", "946713600",
    "--static-network-duration", "3600",
]

# Optional: if stdout from ./replay is heavily buffered and the marker line
# arrives late, you can set this to True (requires `stdbuf` to be installed).
USE_STDBUF = False
STDBUF_PREFIX = ["stdbuf", "-oL", "-eL"]

# Substring that indicates the GPU phase has started.
# Note: "for x iterations" can be any number, so we match the stable prefix.
START_MARKER_SUBSTR = "Starting temporal Monte Carlo simulation for"


def start_mps_daemon():
    """
    Start the NVIDIA MPS control daemon.
    Requires sudo; run this script with appropriate permissions.
    If MPS is already running, this should be harmless.
    """
    print("Starting NVIDIA MPS daemon (sudo nvidia-cuda-mps-control -d)...")
    try:
        subprocess.run(
            ["sudo", "nvidia-cuda-mps-control", "-d"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print("Warning: Failed to start MPS daemon. "
              "Make sure MPS is enabled manually.")
        print(f"Error: {e}")


def run_replay_with_percentage(percentage):
    """
    Run the replay command once with the given CUDA_MPS_ACTIVE_THREAD_PERCENTAGE and measure:
      1) network initialization time (CPU-based; before GPU Monte Carlo starts),
      2) GPU (Monte Carlo) time (after the start marker appears),
      3) total time.

    Returns:
        dict | None
        {
            "network_init": float | None,
            "gpu_time": float | None,
            "total_time": float,
        }
        or None on failure.
    """
    print(f"\n=== Running with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={percentage} ===")

    # Inherit current environment and add/override the CUDA vars
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(percentage)

    # Build the command (optionally with stdbuf to force line-buffered output)
    cmd = (STDBUF_PREFIX + REPLAY_CMD) if USE_STDBUF else REPLAY_CMD

    # Launch process and start timers
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,     # decode to str
        bufsize=1,     # line-buffered in text mode
    )

    t1 = None  # time when we first see the GPU-start marker
    try:
        # Read stdout line by line and forward it to our stdout so we preserve
        # the original behavior of showing the child's output in real-time.
        for line in proc.stdout:
            if line is None:
                continue
            line = line.rstrip("\n")
            print(line)
            # Simple substring match; no regex.
            if t1 is None and START_MARKER_SUBSTR in line:
                t1 = time.perf_counter()
    except Exception as e:
        print(f"Error while reading output: {e}")

    # Wait for process to finish to capture its return code and end time
    return_code = proc.wait()
    t2 = time.perf_counter()

    if return_code != 0:
        elapsed = t2 - t0
        print(f"Run failed with return code {return_code} after {elapsed:.3f} s")
        return None

    # Compute timings
    total_time = t2 - t0
    if t1 is None:
        print(f"Warning: did not see start marker '{START_MARKER_SUBSTR}' in output.")
        print(f"Total elapsed time: {total_time:.3f} s (no breakdown available)")
        return {
            "network_init": None,
            "gpu_time": None,
            "total_time": total_time,
        }

    network_init = t1 - t0
    gpu_time = t2 - t1

    # Preserve the spirit of the original per-run printout while adding detail
    print(f"Network init time: {network_init:.3f} s")
    print(f"GPU (Monte Carlo) time: {gpu_time:.3f} s")
    print(f"Total elapsed time: {total_time:.3f} s")

    return {
        "network_init": network_init,
        "gpu_time": gpu_time,
        "total_time": total_time,
    }


def fmt_or_na(x, width=10, prec=3):
    if x is None:
        return f"{'N/A':>{width}}"
    return f"{x:>{width}.{prec}f}"


def main():
    start_mps_daemon()

    # Optional: warm-up run (unmeasured in results) to JIT/initialize things
    print("Performing warm-up run (not recorded)...")
    _ = run_replay_with_percentage(100)

    # Store detailed results as (percentage, run_index, network_init, gpu_time, total_time)
    results = []

    for p in PERCENTAGES:
        for run_idx in range(1, RUNS_PER_PERCENTAGE + 1):
            print(f"\n--- Percentage {p}%, run {run_idx}/{RUNS_PER_PERCENTAGE} ---")
            res = run_replay_with_percentage(p)
            if res is not None:
                results.append(
                    (p, run_idx, res["network_init"], res["gpu_time"], res["total_time"])
                )

    # === Print results: original-style table (totals only) ===
    print("\n=== Strong-scaling results (all runs; total times) ===")
    print(f"{'Percentage':>12} | {'Run':>3} | {'Time (s)':>10}")
    print("-" * 33)
    for p, run_idx, _, _, t_total in results:
        print(f"{p:>12} | {run_idx:>3} | {t_total:>10.3f}")

    # === Print results: detailed breakdown ===
    print("\n=== Detailed breakdown (all runs) ===")
    print(f"{'Percentage':>12} | {'Run':>3} | {'Network (s)':>12} | {'GPU (s)':>10} | {'Total (s)':>10}")
    print("-" * 66)
    for p, run_idx, t_init, t_gpu, t_total in results:
        print(
            f"{p:>12} | {run_idx:>3} | {fmt_or_na(t_init, 12)} | "
            f"{fmt_or_na(t_gpu, 10)} | {t_total:>10.3f}"
        )

    # === CSV output ===
    # 1) Preserve original CSV (compatibility): one row per run with *total* time only.
    with open("strong_scaling_results.csv", "w") as f:
        f.write("percentage,run_index,time_s\n")
        for p, run_idx, _, _, t_total in results:
            f.write(f"{p},{run_idx},{t_total:.6f}\n")

    # 2) New detailed CSV with breakdown columns.
    with open("strong_scaling_results_detailed.csv", "w") as f:
        f.write("percentage,run_index,network_init_s,gpu_time_s,total_time_s\n")
        for p, run_idx, t_init, t_gpu, t_total in results:
            init_str = "" if t_init is None else f"{t_init:.6f}"
            gpu_str = "" if t_gpu is None else f"{t_gpu:.6f}"
            f.write(f"{p},{run_idx},{init_str},{gpu_str},{t_total:.6f}\n")


if __name__ == "__main__":
    main()
