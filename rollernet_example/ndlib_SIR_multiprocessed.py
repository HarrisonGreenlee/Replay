# run_ndlib_dynamic_sir.py
import math, random, csv, time, argparse, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import dynetx as dn
import ndlib.models.ModelConfig as mc
import ndlib.models.dynamic as dm

# ---- Fill these from your Tacoma/formatter config ----
P_C = 0.0184
STEP_SECONDS = 300.0
INFECTIOUS_SECONDS = 14 * 24 * 3600
N0 = 75
SNAP_PATH = "snapshots_5min.edgelist"
SNAP_MINUTES = 5.0
# ------------------------------------------------------


def read_snapshot_ids(path: str) -> list[int]:
    ids, bad = set(), 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 3:
                bad += 1
                continue
            try:
                ids.add(int(parts[2]))
            except ValueError:
                bad += 1
    if bad:
        print(f"[warn] skipped {bad} malformed lines")
    return sorted(ids)


def _run_one_iteration(
    iteration: int,
    seed: int,
    beta_snap: float,
    gamma_snap: float,
    dt_snap: float,
    n0: int,
    snap_path: str,
):
    """
    Worker process: runs one DynSIR iteration and returns a dict with all series.
    NOTE: Each worker loads the dynamic graph from disk to avoid pickling issues.
    """
    print(f"[iter {iteration}] starting (seed={seed})", flush=True)


    # Per-iteration RNG seeding (reproducible)
    random.seed(seed)
    np.random.seed(seed)

    if not Path(snap_path).exists():
        raise FileNotFoundError(snap_path)

    # Load dynamic graph (read-only in this process)
    G = dn.read_snapshots(snap_path, nodetype=int, timestamptype=int)
    nodes_all = sorted(G.nodes())  # sorted for reproducibility across processes

    if not nodes_all:
        raise RuntimeError("Dynamic graph has no nodes. Check snapshot file.")
    if n0 <= 0:
        raise ValueError("N0 must be > 0 for SIR seeding.")

    # Choose initial infected set deterministically under this seed
    initial_infected = set(random.sample(nodes_all, min(n0, len(nodes_all))))

    # Build and initialize the SIR model
    model = dm.DynSIRModel(G)
    cfg = mc.Configuration()
    cfg.add_model_parameter("beta", beta_snap)
    cfg.add_model_parameter("gamma", gamma_snap)
    cfg.add_model_initial_configuration("Infected", initial_infected)
    model.set_initial_status(cfg)

    # Execute: one iteration per available snapshot
    its = model.execute_snapshots()

    # Maintain a fixed-universe status map across snapshots (S=0, I=1, R=2)
    master = {u: 0 for u in nodes_all}
    for u in initial_infected:
        master[u] = 1

    S_series = [sum(s == 0 for s in master.values())]
    I_series = [sum(s == 1 for s in master.values())]
    R_series = [sum(s == 2 for s in master.values())]

    for it in its:
        for u, st in it.get("status", {}).items():
            master[u] = st
        S_series.append(sum(s == 0 for s in master.values()))
        I_series.append(sum(s == 1 for s in master.values()))
        R_series.append(sum(s == 2 for s in master.values()))

    t_seconds = [k * dt_snap for k in range(len(its) + 1)]

    # Lightweight summary (returned for optional logging)
    peak_I = max(I_series)
    peak_idx = I_series.index(peak_I)
    summary = {
        "steps": len(its),
        "peak_I": peak_I,
        "peak_t": t_seconds[peak_idx],
        "end": (S_series[-1], I_series[-1], R_series[-1]),
    }

    return {
        "iteration": iteration,
        "seed": seed,
        "t_seconds": t_seconds,
        "S": S_series,
        "I": I_series,
        "R": R_series,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run NDlib dynamic SIR over snapshots with multiprocessing."
    )
    parser.add_argument(
        "-n",
        "--iters",
        type=int,
        default=100,
        help="Number of unique iterations (simulations) to run.",
    )
    parser.add_argument(
        "-processes",
        "--processes",
        type=int,
        #default=max(1, (os.cpu_count() or 1) // 2),
        default=10, # since we are critical of the performance of this library it would be good 
                    # to have a number of processes that divides iterations evenly
                    # want to present this library at its best performance
        help="Number of worker processes to use.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="Base seed. Iteration i uses (seed_base + i).",
    )
    parser.add_argument(
        "--out",
        default="out_ndlib.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--snap",
        default=SNAP_PATH,
        help="Path to snapshots edgelist file.",
    )
    parser.add_argument(
        "--snap-minutes",
        type=float,
        default=SNAP_MINUTES,
        help="Snapshot window size in minutes; must match formatter output.",
    )
    args = parser.parse_args()

    # Sanity
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")
    if args.processes <= 0:
        raise ValueError("--processes must be > 0")
    if not Path(args.snap).exists():
        raise FileNotFoundError(args.snap)

    # Convert continuous-time rates to per-snapshot probabilities
    lam = -math.log1p(-P_C) / STEP_SECONDS  # contact hazard rate
    rho = 1.0 / INFECTIOUS_SECONDS          # recovery rate
    dt_snap = args.snap_minutes * 60.0
    beta_snap = 1.0 - math.exp(-lam * dt_snap)
    gamma_snap = 1.0 - math.exp(-rho * dt_snap)

    # Single pass to warn on snapshot ID gaps (time axis is consecutive steps)
    snap_ids = read_snapshot_ids(args.snap)
    if snap_ids and any(b - a > 1 for a, b in zip(snap_ids, snap_ids[1:])):
        print(
            "[note] snapshot ID gaps detected; saving consecutive steps so simulation "
            "time matches updates. Consider emitting empty windows in the source."
        )

    # Prepare seeds and iterations
    seeds = [args.seed_base + i for i in range(1, args.iters + 1)]
    iterations = list(range(1, args.iters + 1))

    start = time.perf_counter()

    results = []
    # Run iterations in parallel
    with ProcessPoolExecutor(max_workers=args.processes) as ex:
        futures = [
            ex.submit(
                _run_one_iteration,
                iteration=it,
                seed=seed,
                beta_snap=beta_snap,
                gamma_snap=gamma_snap,
                dt_snap=dt_snap,
                n0=N0,
                snap_path=args.snap,
            )
            for it, seed in zip(iterations, seeds)
        ]

        for fut in as_completed(futures):
            results.append(fut.result())

    # Sort by iteration for deterministic output
    results.sort(key=lambda r: r["iteration"])

    # Write combined CSV: one row per time point per iteration
    out_rows = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "simulation_id", "infectious", "resistant", "susceptible"])
        for res in results:
            for t, S, I, R in zip(res["t_seconds"], res["S"], res["I"], res["R"]):
                w.writerow([t, res["iteration"], I, R, S])
                out_rows += 1

    elapsed = time.perf_counter() - start

    print(
        f"Saved {out_rows} rows from {args.iters} iterations "
        f"to '{args.out}'. Elapsed {elapsed:.2f}s using {args.processes} processes."
    )


if __name__ == "__main__":
    main()
