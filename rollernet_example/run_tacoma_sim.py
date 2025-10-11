# pip install tacoma pandas matplotlib numpy

import argparse
import math
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import tacoma as tc
import matplotlib.pyplot as plt
# --- NEW: timer
import time

# ---------- IO helpers ----------

def load_edge_changes_from_taco(path: str) -> tc.edge_changes:
    with open(path, "r", encoding="utf-8") as f:
        tn = tc.read_json_taco(f)
    tc.verify(tn)
    return tn

def load_edge_changes_from_txt(path: str, id_base: str = "one") -> tc.edge_changes:
    """
    Load whitespace-delimited file with columns: u v start_ts end_ts
    and convert to tacoma.edge_changes (undirected).
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, :4]
    df.columns = ["u", "v", "start_ts", "end_ts"]

    # IDs -> 0-based for Tacoma
    if id_base.lower().startswith("one"):
        df["u"] = df["u"].astype(int) - 1
        df["v"] = df["v"].astype(int) - 1
    else:
        df["u"] = df["u"].astype(int)
        df["v"] = df["v"].astype(int)

    # Build edge_changes
    evmap = {}
    for u, v, s, e in df[["u", "v", "start_ts", "end_ts"]].itertuples(index=False):
        if u > v:
            u, v = v, u
        s = float(s); e = float(e)
        evmap.setdefault(s, {"in": [], "out": []})
        evmap.setdefault(e, {"in": [], "out": []})
        evmap[s]["in"].append((int(u), int(v)))
        evmap[e]["out"].append((int(u), int(v)))

    times = sorted(evmap.keys())
    nodes = sorted(set(df["u"]).union(set(df["v"])))
    N = max(nodes) + 1 if nodes else 0

    tn = tc.edge_changes()
    tn.N = int(N)
    tn.t0 = float(df["start_ts"].min()) if len(df) else 0.0
    tn.tmax = float(df["end_ts"].max()) if len(df) else 0.0
    tn.t = [float(t) for t in times]
    tn.edges_initial = []
    tn.edges_in  = [evmap[t]["in"]  for t in times]
    tn.edges_out = [evmap[t]["out"] for t in times]
    tn.time_unit = "s"
    tn.notes = "Constructed from (u,v,start,end) file."
    tc.verify(tn)
    return tn

# ---------- Parameter mapping (probability per step -> rates) ----------

def prob_per_step_to_rate(p: float, dt: float) -> float:
    """
    Convert per-time-step probability p (over duration dt seconds) to a
    continuous-time rate lambda via: 1 - exp(-lambda * dt) = p  => lambda = -ln(1-p)/dt.
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        # practically infinite; cap to a large rate to avoid inf
        return 1e9
    return -math.log(1.0 - p) / dt

# ---------- Simulation runners ----------

def run_sir(tn: tc.edge_changes, p_c: float, step_seconds: float, infectious_period_s: float,
            n0: int = 1, seed: int = 42, t_multiplier: float = 1.0):
    """
    SIR on temporal network with Tacoma's Gillespie SSA.
    """
    beta = prob_per_step_to_rate(p_c, step_seconds)
    rho  = 1.0 / infectious_period_s
    t_sim = max(1e-6, (tn.tmax - tn.t0) * t_multiplier)
    sir = tc.SIR(tn.N, t_sim, beta, rho,
                 number_of_initially_infected=n0,
                 seed=seed)
    tc.gillespie_SIR(tn, sir)
    return sir, {"beta": beta, "rho": rho, "t_sim": t_sim}

def run_sis(tn: tc.edge_changes, p_c: float, step_seconds: float, infectious_period_s: float,
            n0: int = 1, seed: int = 43, t_multiplier: float = 1.0):
    beta = prob_per_step_to_rate(p_c, step_seconds)
    rho  = 1.0 / infectious_period_s
    t_sim = max(1e-6, (tn.tmax - tn.t0) * t_multiplier)
    sis = tc.SIS(tn.N, t_sim, beta, rho,
                 number_of_initially_infected=n0,
                 seed=seed)
    tc.gillespie_SIS(tn, sis)
    return sis, {"beta": beta, "rho": rho, "t_sim": t_sim}

# ---------- Ensemble helpers ----------

def _resample_step_fn(t_src, y_src, t_grid):
    """Right-continuous step interpolation onto a common grid."""
    out = []
    j = 0
    last = y_src[0] if y_src else 0
    n = len(t_src)
    for tg in t_grid:
        while j < n and t_src[j] <= tg:
            last = y_src[j]
            j += 1
        out.append(last)
    return out

def ensemble_quantiles(tn, model="SIR", runs=200, seed0=123,
                       p_c=0.1, step_seconds=300.0,
                       infectious_seconds=14*24*3600,
                       n0=75, t_multiplier=1.0,
                       qs=(0.05, 0.5, 0.95)):
    """
    Run many realizations and return:
      - t_grid: unified time grid (starts at 0)
      - q: dict of pointwise quantiles for S,I,(R)
      - qs: the quantile tuple used
      - stacks: dict of resampled per-run trajectories; each value is
                an array with shape (runs, len(t_grid))
      - t0_abs: absolute time corresponding to t_grid[0]
    """
    trajs = []
    for k in range(runs):
        if model == "SIR":
            epi, _ = run_sir(tn, p_c, step_seconds, infectious_seconds,
                             n0=n0, seed=seed0 + k, t_multiplier=t_multiplier)
            S = [tn.N - i - r for i, r in zip(epi.I, epi.R)]
            trajs.append({"t": epi.time, "S": S, "I": epi.I, "R": epi.R})
        else:
            epi, _ = run_sis(tn, p_c, step_seconds, infectious_seconds,
                             n0=n0, seed=seed0 + k, t_multiplier=t_multiplier)
            S = [tn.N - i for i in epi.I]
            trajs.append({"t": epi.time, "S": S, "I": epi.I})

    if not trajs or not trajs[0]["t"]:
        return [], {}, qs, {}, None  # --- adjusted to include t0_abs=None

    # Common grid: all unique event times, shifted so time starts at 0
    all_times = sorted({tt for tr in trajs for tt in tr["t"]})
    t0 = all_times[0]
    t_grid = [tt - t0 for tt in all_times]

    # Stack each series
    keys = ["S", "I"] + (["R"] if model == "SIR" else [])
    stacks = {k: [] for k in keys}
    for tr in trajs:
        tshift = [tt - t0 for tt in tr["t"]]
        for k in keys:
            stacks[k].append(_resample_step_fn(tshift, tr[k], t_grid))

    # Convert lists to arrays (runs, T)
    for k in list(stacks.keys()):
        stacks[k] = np.asarray(stacks[k], dtype=float)

    # Pointwise quantiles (len(qs), T)
    q = {}
    for k in keys:
        q[k] = np.quantile(stacks[k], qs, axis=0)

    return t_grid, q, qs, stacks, t0  # --- adjusted to also return absolute t0

def plot_bands_and_trajs(t_grid, q, qs, stacks, tn, model="SIR",
                         show_S=True, show_trajs=False):
    """
    Plot median + (qlo,qhi) ribbon for I (and R), plus S median.
    If show_trajs=True, overlay each trajectory as a thin, light line
    using consistent colors for S, I, R.
    """
    # Consistent colors for each compartment
    colors = {
        "S": "tab:blue",
        "I": "tab:orange",
        "R": "tab:green"
    }

    qlo_idx, qmed_idx, qhi_idx = 0, 1, 2

    # Helper to draw traj stack
    def _plot_trajs(var_key, label):
        if var_key not in stacks:
            return
        # Each row is a trajectory
        for row in stacks[var_key]:
            plt.step(t_grid, row, where="post",
                     linewidth=0.6, alpha=0.08, color=colors[var_key])
        # Put a light legend hint for the trajectories
        # (we rely on the median/ribbon for the main legend entries)

    # Optionally draw trajectories first (so ribbons/medians are on top)
    if show_trajs:
        if show_S and "S" in stacks:
            _plot_trajs("S", "S trajectories")
        _plot_trajs("I", "I trajectories")
        if model == "SIR" and "R" in stacks:
            _plot_trajs("R", "R trajectories")

    # Draw ribbons + medians
    # I
    i_lo, i_med, i_hi = q["I"][qlo_idx], q["I"][qmed_idx], q["I"][qhi_idx]
    plt.fill_between(t_grid, i_lo, i_hi, alpha=0.25, step="post",
                     label=f"I {int((qs[2]-qs[0])*100)}% band",
                     color=colors["I"])
    plt.step(t_grid, i_med, where="post", label="I median", color=colors["I"])

    # R (SIR only)
    if "R" in q:
        r_lo, r_med, r_hi = q["R"][qlo_idx], q["R"][qmed_idx], q["R"][qhi_idx]
        plt.fill_between(t_grid, r_lo, r_hi, alpha=0.15, step="post",
                         label="R band", color=colors["R"])
        plt.step(t_grid, r_med, where="post", label="R median", color=colors["R"])

    # S median for reference
    if show_S and "S" in q:
        s_med = q["S"][qmed_idx]
        plt.step(t_grid, s_med, where="post", label="S median", color=colors["S"])

    plt.xlabel(f"time since start [{tn.time_unit}]")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()

# --- NEW: CSV writer for long per-simulation trajectories (S, I, [R]) ---
def _write_per_simulation_csv(out_prefix: str,
                              model: str,
                              t_abs,
                              stacks: dict,
                              tn: tc.edge_changes):
    """
    Write one long CSV with columns:
      time, simulation_id, infectious, resistant, susceptible
    Using absolute timestamps (same unit as the input network, typically seconds).
    Returns (n_rows_written, csv_path).
    """
    if not out_prefix or "I" not in stacks:
        return 0, ""
    t_abs = np.asarray(t_abs, dtype=float)
    runs = int(stacks["I"].shape[0])
    T = t_abs.shape[0]

    rows = []
    for sim_id in range(runs):
        infectious = np.asarray(stacks["I"][sim_id], dtype=float)
        if "R" in stacks:
            resistant = np.asarray(stacks["R"][sim_id], dtype=float)
        else:
            resistant = np.zeros(T, dtype=float)
        if "S" in stacks:
            susceptible = np.asarray(stacks["S"][sim_id], dtype=float)
        else:
            susceptible = tn.N - infectious - resistant

        df_sim = pd.DataFrame({
            "time": t_abs,
            "simulation_id": sim_id,
            "infectious": infectious,
            "resistant": resistant,
            "susceptible": susceptible
        })
        rows.append(df_sim)

    df_long = pd.concat(rows, ignore_index=True)
    csv_path_traj = f"{out_prefix}_trajectories_{model}.csv"
    df_long.to_csv(csv_path_traj, index=False)
    print(f"Wrote per-simulation trajectories to {csv_path_traj}")
    return len(df_long), csv_path_traj

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Run Tacoma epidemic on a temporal network, with optional ensemble bands.")
    ap.add_argument("--taco", help="Path to .taco (Tacoma JSON). If omitted, use --txt.")
    ap.add_argument("--txt",  help="Path to whitespace (u v start end) file.")
    ap.add_argument("--id-base", choices=["one","zero"], default="one", help="Node ID base in --txt (default: one).")
    ap.add_argument("--model", choices=["SIR","SIS"], default="SIR", help="Epidemic model.")
    ap.add_argument("--p_c", type=float, default=0.2, help="Infection probability per step (your model).")
    ap.add_argument("--step-seconds", type=float, default=3600.0, help="Step length Δt (seconds) used for p_c.")
    ap.add_argument("--infectious-seconds", type=float, default=14*24*3600, help="Infectious period (seconds).")
    ap.add_argument("--n0", type=int, default=75, help="Initially infected count.")
    ap.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    ap.add_argument("--t-mult", type=float, default=1.0, help="Multiply observation window (loops the network).")
    ap.add_argument("--plot", action="store_true", help="Plot a single-run trajectory.")
    # Ensemble options
    ap.add_argument("--runs", type=int, default=100, help="Number of stochastic runs (>=1).")
    ap.add_argument("--bands", action="store_true", help="If set and runs>1, plot median and quantile ribbons.")
    ap.add_argument("--qlo", type=float, default=0.05, help="Lower quantile for bands (e.g., 0.05).")
    ap.add_argument("--qhi", type=float, default=0.95, help="Upper quantile for bands (e.g., 0.95).")
    ap.add_argument("--out-prefix", type=str, default="out_tacoma.csv", help="If set, write CSVs with quantile series to this prefix.")
    ap.add_argument("--show-trajs", action="store_true",
                    help="Overlay all ensemble trajectories as thin, light lines.")
    # --- NEW: aliases + processes for summary line
    ap.add_argument("--iters", type=int, help="Alias for --runs; if set, overrides --runs.")
    ap.add_argument("--out", type=str, default="", help="Alias for --out-prefix; if set, overrides --out-prefix.")

    args = ap.parse_args()

    # --- NEW: start timer
    t_start = time.perf_counter()

    # --- NEW: map aliases so the rest of the code remains unchanged
    if args.iters is not None:
        args.runs = args.iters
    else:
        args.iters = args.runs  # ensure available for the final print
    if args.out:
        args.out_prefix = args.out
    else:
        args.out = args.out_prefix  # ensure available for the final print

    # Track rows written + path for the final summary line
    # --- NEW:
    out_rows = 0
    traj_csv_path_for_timer = ""

    # Load network
    if args.taco:
        tn = load_edge_changes_from_taco(args.taco)
    elif args.txt:
        tn = load_edge_changes_from_txt(args.txt, id_base=args.id_base)
    else:
        raise SystemExit("Provide either --taco or --txt.")

    # Single run (for logging and optional simple plot)
    if args.model == "SIR":
        epi, pars = run_sir(tn, args.p_c, args.step_seconds, args.infectious_seconds,
                            n0=args.n0, seed=args.seed, t_multiplier=args.t_mult)
        print(f"[SIR] beta={pars['beta']:.6g}  rho={pars['rho']:.6g}  t_sim={pars['t_sim']:.1f}s")
        if epi.I:
            print(f"Peak I: {max(epi.I)} at t≈{epi.time[epi.I.index(max(epi.I))]:.1f}s; Final R={epi.R[-1]}")
    else:
        epi, pars = run_sis(tn, args.p_c, args.step_seconds, args.infectious_seconds,
                            n0=args.n0, seed=args.seed, t_multiplier=args.t_mult)
        print(f"[SIS] beta={pars['beta']:.6g}  rho={pars['rho']:.6g}  t_sim={pars['t_sim']:.1f}s")
        if epi.I:
            print(f"Peak I: {max(epi.I)} at t≈{epi.time[epi.I.index(max(epi.I))]:.1f}s; Final I={epi.I[-1]}")

    did_plot_anything = False

    # Optional single-run plot
    if args.plot:
        did_plot_anything = True
        # time axis starting at 0 for readability
        t0 = epi.time[0] if epi.time else 0.0
        t = [tt - t0 for tt in epi.time]

        if args.model == "SIR":
            S = [tn.N - i - r for i, r in zip(epi.I, epi.R)]
            plt.step(t, S, where="post", label="S (single run)")
            plt.step(t, epi.I, where="post", label="I (single run)")
            plt.step(t, epi.R, where="post", label="R (single run)")
        else:  # SIS
            S = [tn.N - i for i in epi.I]
            plt.step(t, S, where="post", label="S (single run)")
            plt.step(t, epi.I, where="post", label="I (single run)")

    # --- NEW: write per-simulation CSV even for a single run (simulation_id=0)
    if args.out_prefix and args.runs <= 1 and epi.I:
        if args.model == "SIR":
            S_single = np.asarray([[tn.N - i - r for i, r in zip(epi.I, epi.R)]], dtype=float)
            stacks_single = {
                "I": np.asarray([epi.I], dtype=float),
                "R": np.asarray([epi.R], dtype=float),
                "S": S_single
            }
        else:  # SIS
            S_single = np.asarray([[tn.N - i for i in epi.I]], dtype=float)
            stacks_single = {
                "I": np.asarray([epi.I], dtype=float),
                "S": S_single
            }
        t_abs_single = np.asarray(epi.time, dtype=float)  # absolute timestamps from the simulator
        rows, path = _write_per_simulation_csv(args.out_prefix, args.model, t_abs_single, stacks_single, tn)
        out_rows += rows
        if not traj_csv_path_for_timer:
            traj_csv_path_for_timer = path

    # Ensemble ribbons (and optional trajectories)
    if args.runs > 1:
        qlo, qhi = float(args.qlo), float(args.qhi)
        if not (0.0 <= qlo < qhi <= 1.0):
            raise SystemExit("Require 0 <= --qlo < --qhi <= 1.")
        qs = (qlo, 0.5, qhi)
        t_grid, q, _, stacks, t0_abs = ensemble_quantiles(
            tn,
            model=args.model,
            runs=args.runs,
            seed0=args.seed,  # reproducible wrt base seed
            p_c=args.p_c,
            step_seconds=args.step_seconds,
            infectious_seconds=args.infectious_seconds,
            n0=args.n0,
            t_multiplier=args.t_mult,
            qs=qs
        )

        # --- NEW: write long per-simulation trajectories like your example ---
        if args.out_prefix and t_grid and t0_abs is not None and "I" in stacks:
            t_abs = np.asarray(t_grid, dtype=float) + float(t0_abs)
            rows, path = _write_per_simulation_csv(args.out_prefix, args.model, t_abs, stacks, tn)
            out_rows += rows
            if not traj_csv_path_for_timer:
                traj_csv_path_for_timer = path

        # Save CSVs if requested (quantile bands)
        if args.out_prefix and t_grid:
            df = pd.DataFrame({"t": t_grid})
            for k in q.keys():
                df[f"{k}_q{int(100*qlo)}"] = q[k][0]
                df[f"{k}_q50"] = q[k][1]
                df[f"{k}_q{int(100*qhi)}"] = q[k][2]
            csv_path = f"{args.out_prefix}_bands_{args.model}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Wrote quantile bands to {csv_path}")

        if args.bands and t_grid:
            did_plot_anything = True
            plot_bands_and_trajs(
                t_grid, q, qs, stacks, tn,
                model=args.model, show_S=True, show_trajs=args.show_trajs
            )

    if did_plot_anything:
        plt.show()

    # --- NEW: final runtime summary
    elapsed = time.perf_counter() - t_start
    # ensure args.out points to the trajectories file if available
    if not args.out and traj_csv_path_for_timer:
        args.out = traj_csv_path_for_timer

    print(
        f"Saved {out_rows} rows from {args.iters} iterations "
        f"to '{args.out}'. Elapsed {elapsed:.2f}s."
    )

if __name__ == "__main__":
    main()
