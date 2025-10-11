#!/usr/bin/env python3
import subprocess
import time
import math
import os
import csv
import hashlib
from datetime import datetime


SCENARIOS = {
    "airport": {
        "group_multiplier": 0.01,
        "prob_inner": 0.9,
        "prob_inter": 0.5,
        "output_csv": "airport.csv"
    },
    "villages": {
        "group_multiplier": 0.1,
        "prob_inner": 0.9,
        "prob_inter": 0.05,
        "output_csv": "villages.csv"
    },
    "cities": {
        "group_multiplier": 0.01,
        "prob_inner": 0.9,
        "prob_inter": 0.01,
        "output_csv": "cities.csv"
    },
    "dense": {
        "group_multiplier": 1.0,
        "prob_inner": 1,
        "prob_inter": 1,
        "output_csv": "dense.csv"
    },
}

POPULATION_SIZES = [
    25, 50, 75, 100, 200, 300, 400, 500,
    600, 700, 800, 900, 
    1000, 
    1500, 2000,
    2500, 5000, 10000
]

START_DATETIME = "2000-01-01T00:00:00"
START_TIME_EPOCH = int(datetime.fromisoformat(START_DATETIME).timestamp())
STEPS = 100
STEP_SIZE_SECONDS = 3600
SIM_STEP_SIZE = float(STEP_SIZE_SECONDS)
SIM_ITERATIONS = STEPS
SIM_PARALLEL_RUNS = 10000
MAX_RUNTIME = 10 * 60  # 10 minutes


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

def read_num_lines_from_file(filename):
    num_lines = sum(1 for _ in open(filename))
    return num_lines

def deterministic_seed(*args):
    s = "_".join(map(str, args))
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (10**8)


def main():
    compute_types = {
        "cpu": "--cpu-only",
        "gpu": ""
    }

    for scenario_name, config in SCENARIOS.items():
        csv_file = config["output_csv"]
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            header = ["population", "steps", "num_edges", "network_seed"] + [f"{ctype}_runtime" for ctype in compute_types]
            writer.writerow(header)

            skip_map = {ctype: False for ctype in compute_types}

            for pop_size in POPULATION_SIZES:
                if all(skip_map.values()):
                    print(f"[INFO] Skipping population={pop_size} for {scenario_name} — all compute types have failed.")
                    writer.writerow([pop_size, STEPS, math.nan, math.nan] + [math.nan] * len(compute_types))
                    continue

                num_groups = max(1, int(config["group_multiplier"] * pop_size))
                graph_file = f"{scenario_name}_{pop_size}.txt"
                network_seed = deterministic_seed(scenario_name, pop_size)

                gen_cmd = (
                    f"contact_network.exe \"{START_DATETIME}\" "
                    f"{STEPS} {STEP_SIZE_SECONDS} "
                    f"{pop_size} {num_groups} "
                    f"{config['prob_inner']} {config['prob_inter']} "
                    f"{graph_file} {network_seed}"
                )

                print(f"[INFO] Generating graph for {scenario_name}, population={pop_size}...")
                gen_time = run_subprocess(gen_cmd)
                if gen_time is None:
                    print(f"[WARNING] Generation of {graph_file} failed or timed out.")
                    writer.writerow([pop_size, STEPS, math.nan, network_seed] + [math.nan] * len(compute_types))
                    continue

                try:
                    # num_edges = read_num_edges_from_file(graph_file)
                    num_edges = read_num_lines_from_file(graph_file) - 1 # subtract 1 for header file
                    print(f"[INFO] Read {num_edges:,} edges from file.")
                except Exception as e:
                    print(f"[ERROR] Could not read number of edges: {e}")
                    writer.writerow([pop_size, STEPS, math.nan, network_seed] + [math.nan] * len(compute_types))
                    continue

                run_times = {}

                for ctype, ctype_args in compute_types.items():
                    if skip_map[ctype]:
                        run_times[ctype] = math.nan
                        continue

                    sim_cmd = (
                        f"replay {graph_file} "
                        f"{ctype_args} "
                        f"--step-size {SIM_STEP_SIZE} "
                        f"--iterations {SIM_ITERATIONS} "
                        f"--N {pop_size} "
                        f"--M {SIM_PARALLEL_RUNS} "
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

                    print(f"[INFO] Running {ctype.upper()} simulation for {scenario_name}, population={pop_size}...")
                    sim_time = run_subprocess(sim_cmd)

                    if sim_time is None:
                        print(f"[WARNING] {ctype.upper()} simulation for population={pop_size} failed.")
                        skip_map[ctype] = True
                        run_times[ctype] = math.nan
                    else:
                        run_times[ctype] = sim_time
                        print(f"[INFO] {ctype.upper()} simulation completed in {sim_time:.2f} seconds.")

                row = [pop_size, STEPS, num_edges, network_seed] + [run_times[ctype] for ctype in compute_types]
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())

                try:
                    os.remove(graph_file)
                    print(f"[INFO] Deleted temporary file: {graph_file}")
                except Exception as e:
                    print(f"[WARNING] Could not delete {graph_file}: {e}")

            print(f"[INFO] Results stored in {csv_file}.")

if __name__ == "__main__":
    main()
