# #!/usr/bin/env python3
# import subprocess
# import time
# import math
# import csv
# import os
# import hashlib
# from datetime import datetime

# def deterministic_seed(*args):
#     s = "_".join(map(str, args))
#     return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (10**8)

# def run_subprocess(cmd, timeout=300):
#     start = time.time()
#     try:
#         subprocess.run(cmd, shell=True, check=True, timeout=timeout)
#     except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
#         return None
#     return time.time() - start

# def read_num_edges_from_file(filename):
#     with open(filename, "r") as f:
#         for line in f:
#             if line.startswith("NUM_EDGES"):
#                 return int(line.strip().split()[1])
#     raise ValueError("NUM_EDGES line not found in the file.")

# ###############################
# # PARAMETERS
# ###############################

# NUM_NODES = 1000
# GROUP_MULTIPLIER = 1.0  # Each node is its own group (dense)
# P_INNER = 1.0           # No effect since groups are of size 1
# STEP_SIZE_SECONDS = 3600
# SIM_STEP_SIZE = float(STEP_SIZE_SECONDS)
# STEPS = 100
# START_DATETIME = "2000-01-01T00:00:00"
# START_TIME_EPOCH = int(datetime.fromisoformat(START_DATETIME).timestamp())
# MAX_RUNTIME = 5 * 60  # 5 minutes

# graph_file = "temp.txt"
# csv_output = "3d_runtime_surface.csv"

# # Values to explore
# P_INTER_VALUES = [i * 0.01 for i in range(101)]
# PARALLEL_SIM_RUNS = [50000 * i for i in range(42)]

# ###############################
# # MAIN GRID EXPLORATION
# ###############################

# def main():
#     with open(csv_output, mode="w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["num_edges", "num_parallel_simulations", "seed", "steps", "cpu_runtime_seconds", "gpu_runtime_seconds"])

#         for p_inter in P_INTER_VALUES:
#             seed = deterministic_seed(NUM_NODES, p_inter)

#             # Generate the contact network
#             gen_cmd = (
#                 f"contact_network.exe \"{START_DATETIME}\" {STEPS} {STEP_SIZE_SECONDS} "
#                 f"{NUM_NODES} {NUM_NODES} "
#                 f"{P_INNER} {p_inter} "
#                 f"{graph_file} {seed}"
#             )

#             print(f"[INFO] Generating graph with p_inter={p_inter}...")
#             gen_time = run_subprocess(gen_cmd)
#             if gen_time is None:
#                 print(f"[ERROR] Graph generation failed at p_inter={p_inter}.")
#                 continue

#             try:
#                 total_edges = read_num_edges_from_file(graph_file)
#                 print(f"[INFO] Read {total_edges:,} edges from file.")
#             except Exception as e:
#                 print(f"[ERROR] Could not read number of edges: {e}")
#                 continue

#             for M in PARALLEL_SIM_RUNS:
#                 runtimes = {}
#                 for mode, extra_args in {"cpu": "--cpu-only", "gpu": ""}.items():
#                     sim_cmd = (
#                         f"gpu_cpu_temporal_sim {graph_file} "
#                         f"{extra_args} "
#                         f"--step-size {SIM_STEP_SIZE} "
#                         f"--iterations {STEPS} "
#                         f"--N {NUM_NODES} "
#                         f"--M {M} "
#                         f"--initial-infected 0.5 "
#                         f"--infect-prob 1.0 "
#                         f"--upper-range 7200 "
#                         f"--medium-range 3600 "
#                         f"--lower-range 0 "
#                         f"--start-time {START_TIME_EPOCH} "
#                         f"--time-step {STEP_SIZE_SECONDS}"
#                     )

#                     print(f"[INFO] Running {mode.upper()} simulation with M={M} parallel runs...")
#                     sim_time = run_subprocess(sim_cmd)

#                     if sim_time is None:
#                         print(f"[WARN] {mode.upper()} simulation failed for M={M}, p_inter={p_inter}.")
#                         sim_time = math.nan

#                     runtimes[mode] = sim_time

#                 writer.writerow([total_edges * STEPS, M, seed, STEPS, runtimes["cpu"], runtimes["gpu"]])
#                 f.flush()
#                 os.fsync(f.fileno())

#             try:
#                 os.remove(graph_file)
#             except Exception as e:
#                 print(f"[WARNING] Could not delete {graph_file}: {e}")

#     print(f"[INFO] 3D runtime surface data written to {csv_output}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import subprocess
import time
import math
import csv
import os
import hashlib
from datetime import datetime

def deterministic_seed(*args):
    s = "_".join(map(str, args))
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (10**8)

def run_subprocess(cmd, timeout=300):
    start = time.time()
    try:
        subprocess.run(cmd, shell=True, check=True, timeout=timeout)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return None
    return time.time() - start

def read_num_edges_from_file(filename):
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("NUM_EDGES"):
                return int(line.strip().split()[1])
    raise ValueError("NUM_EDGES line not found in the file.")

###############################
# PARAMETERS
###############################

NUM_NODES = 1000
P_INNER = 1.0
STEP_SIZE_SECONDS = 3600
SIM_STEP_SIZE = float(STEP_SIZE_SECONDS)
STEPS = 100
START_DATETIME = "2000-01-01T00:00:00"
START_TIME_EPOCH = int(datetime.fromisoformat(START_DATETIME).timestamp())
MAX_RUNTIME = 5 * 60  # 5 minutes

graph_file = "temp.txt"
csv_output = "3d_runtime_surface.csv"

P_INTER_VALUES = [i * 0.1 for i in range(1, 11)]
PARALLEL_SIM_RUNS = [100000 * i for i in range(1, 11)]  

###############################
# MAIN GRID EXPLORATION
###############################

def main():
    with open(csv_output, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "num_edges", "num_parallel_simulations", "seed", "steps",
            "cpu_runtime_seconds", "gpu_runtime_seconds"
        ])

        failed_cpu = set()
        failed_gpu = set()

        total_p_inter = len(P_INTER_VALUES)
        total_M = len(PARALLEL_SIM_RUNS)

        for p_index, p_inter in enumerate(P_INTER_VALUES):
            print(f"\n{'='*20} [{p_index+1}/{total_p_inter}] Starting p_inter = {p_inter:.4f} {'='*20}")
            seed = deterministic_seed(NUM_NODES, p_inter)

            gen_cmd = (
                f"contact_network.exe \"{START_DATETIME}\" {STEPS} {STEP_SIZE_SECONDS} "
                f"{NUM_NODES} {NUM_NODES} "
                f"{P_INNER} {p_inter} "
                f"{graph_file} {seed}"
            )

            gen_time = run_subprocess(gen_cmd)
            if gen_time is None:
                print(f"[ERROR] Graph generation failed at p_inter={p_inter}.")
                continue

            try:
                total_edges = read_num_edges_from_file(graph_file)
                print(f"[INFO] Read {total_edges:,} edges from file.")
            except Exception as e:
                print(f"[ERROR] Could not read number of edges: {e}")
                continue

            for m_index, M in enumerate(PARALLEL_SIM_RUNS):
                print(f"\n----- [{m_index+1}/{total_M}] Running M = {M} -----")
                runtimes = {}

                for mode, extra_args, failure_set in [
                    #("cpu", "--cpu-only", failed_cpu),
                    ("gpu", "", failed_gpu),
                ]:
                    if any(p >= p_inter and m >= M for (p, m) in failure_set):
                        print(f"[SKIP] Skipping {mode.upper()} at p_inter={p_inter}, M={M} due to previous timeout.")
                        runtimes[mode] = math.nan
                        continue

                    sim_cmd = (
                        f"gpu_cpu_temporal_sim {graph_file} "
                        f"{extra_args} "
                        f"--step-size {SIM_STEP_SIZE} "
                        f"--iterations {STEPS} "
                        f"--N {NUM_NODES} "
                        f"--M {M} "
                        f"--initial-infected 0.5 "
                        f"--infect-prob 1.0 "
                        # f"--upper-range 7200 "
                        # f"--medium-range 3600 "
                        # f"--lower-range 0 "
                        f"--exposed-duration 3600 "
                        f"--infectious-duration 3600 "
                        f"--resistant-duration 0 "
                        f"--start-time {START_TIME_EPOCH} "
                        f"--time-step {STEP_SIZE_SECONDS}"
                    )

                    print(f"[INFO] Running {mode.upper()} simulation with M={M}...")
                    sim_time = run_subprocess(sim_cmd, timeout=MAX_RUNTIME)

                    if sim_time is None:
                        print(f"[WARN] {mode.upper()} simulation failed (timeout) for M={M}, p_inter={p_inter}.")
                        failure_set.add((p_inter, M))
                        sim_time = math.nan

                    runtimes[mode] = sim_time

                writer.writerow([
                    total_edges, M, seed, STEPS, # fixed a bug where total_edges was multiplied by STEPS (100)
                    runtimes.get("cpu", math.nan),
                    runtimes.get("gpu", math.nan)
                ])
                f.flush()
                os.fsync(f.fileno())

            try:
                os.remove(graph_file)
            except Exception as e:
                print(f"[WARNING] Could not delete {graph_file}: {e}")

    print(f"\n[INFO] 3D runtime surface data written to {csv_output}")

if __name__ == "__main__":
    main()
