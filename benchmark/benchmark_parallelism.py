#!/usr/bin/env python3
import subprocess
import time
import os
import csv
import math
import hashlib
import datetime

# Config
SCENARIO = "villages"
POPULATION = 1000
SIM_PARALLEL_RUNS_LIST = [
    10, 100, 1000, 10000, 100000, 200000, 300000,
    400000, 500000, 600000, 700000, 800000, 900000, 1000000,
    1100000, 1200000, 1300000 ,1400000, 1500000, 1600000, 1700000, 1800000, 1900000, 2000000  
]
MAX_RUNTIME = 5 * 60  # 5 minutes timeout
START_DATETIME = "2000-01-01T00:00:00"
START_TIME_EPOCH = int(datetime.datetime.fromisoformat(START_DATETIME).timestamp())
STEPS = 100
STEP_SIZE_SECONDS = 3600
SIM_STEP_SIZE = float(STEP_SIZE_SECONDS)
SIM_ITERATIONS = STEPS
CSV_FILE = "parallelism_benchmark.csv"

# Scenario-specific parameters
GROUP_MULTIPLIER = 0.1
PROB_INNER = 0.9
PROB_INTER = 0.05

def deterministic_seed(*args):
    s = "_".join(map(str, args))
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (10**8)

def run_subprocess(cmd, timeout=MAX_RUNTIME):
    start_time = time.time()
    try:
        subprocess.run(cmd, shell=True, check=True, timeout=timeout)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return None
    return time.time() - start_time

def read_num_edges_from_file(filename):
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("NUM_EDGES"):
                return int(line.strip().split()[1])
    raise ValueError("NUM_EDGES line not found in the file.")

def generate_network(population, network_seed, graph_file):
    num_groups = max(1, int(GROUP_MULTIPLIER * population))

    gen_cmd = (
        f"contact_network.exe \"{START_DATETIME}\" {STEPS} {STEP_SIZE_SECONDS} "
        f"{population} {num_groups} {PROB_INNER} {PROB_INTER} {graph_file} {network_seed}"
    )

    print(f"[INFO] Generating contact network for {population} people with seed {network_seed}...")
    gen_time = run_subprocess(gen_cmd)
    if gen_time is None:
        raise RuntimeError("Failed to generate contact network.")
    print(f"[INFO] Network generated in {gen_time:.2f} seconds.")

def run_benchmarks():
    graph_file = f"{SCENARIO}_{POPULATION}_shared.txt"
    network_seed = 42  # Fixed seed since the network is shared across runs

    # Generate the network only once
    generate_network(POPULATION, network_seed, graph_file)

    try:
        total_edges = read_num_edges_from_file(graph_file)
        print(f"[INFO] Read {total_edges:,} edges from file.")
    except Exception as e:
        print(f"[ERROR] Could not read number of edges: {e}")
        return

    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SIM_PARALLEL_RUNS", "steps", "runtime_seconds", "network_seed", "num_edges"])

        for sim_runs in SIM_PARALLEL_RUNS_LIST:
            # Run the simulation using the shared graph
            sim_cmd = (
                f"gpu_cpu_temporal_sim {graph_file} "
                f"--step-size {SIM_STEP_SIZE} "
                f"--iterations {SIM_ITERATIONS} "
                f"--N {POPULATION} "
                f"--M {sim_runs} "
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

            print(f"[INFO] Running GPU sim with M={sim_runs}...")
            sim_time = run_subprocess(sim_cmd)

            if sim_time is None:
                print(f"[WARNING] Simulation with M={sim_runs} timed out or failed.")
                break
            else:
                print(f"[INFO] Completed in {sim_time:.2f} seconds.")
                writer.writerow([sim_runs, STEPS, sim_time, network_seed, total_edges])
                f.flush()
                os.fsync(f.fileno())

    # Clean up the shared graph file
    try:
        os.remove(graph_file)
        print(f"[INFO] Deleted shared temporary file: {graph_file}")
    except Exception as e:
        print(f"[WARNING] Could not delete {graph_file}: {e}")

    print(f"[INFO] Benchmark results saved to {CSV_FILE}")

def main():
    run_benchmarks()

if __name__ == "__main__":
    main()
