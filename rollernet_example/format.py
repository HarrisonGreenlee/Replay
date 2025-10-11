# format_contacts_min.py
# - Reads a whitespace-delimited preprocessed file (u v start_ts end_ts)
# - Emits:
#     (A) EXACT human-readable format:
#           NODE_LIST <ids...>
#           u, v, YYYY-MM-DDTHH:MM:SSZ, YYYY-MM-DDTHH:MM:SSZ;
#           ...
#     (B) A valid tacoma ".taco" file (JSON) describing the same temporal network
#         as an edge_changes object.
#     (C) DyNetX/NDlib snapshot edgelist "u v t" with 5-minute windows
#
# Hard-coded configuration lives at the top of the file; no CLI flags.

from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Iterable, Optional
import math
import pandas as pd
import os, json

# ---------- settings ----------
INPUT_PATH        = "contacts_preprocessed.dat"
OUT_TEXT_PATH     = "contacts_replay.txt"
OUT_TACO_PATH     = "contacts.taco"
SNAPSHOTS_OUT     = "snapshots_5min.edgelist"
EPIHIPER_DIR       = "epihiper_export"
EPIHIPER_TIME_UNIT = "s"


WRITE_REPLAY          = True
WRITE_TACO            = True
WRITE_NDLIB_SNAPSHOTS = True
WRITE_EPIHIPER        = True

# Tacoma expects 0-based node IDs. If your input is 1-based (typical), set to "one".
ID_BASE_FOR_TACOMA = "one"   # {"one", "zero"}

# --- NEW: Remap IDs to a compact 0..M-1 only for the .taco output
SEQUENTIALIZE_TACO_NODE_IDS = True

# NDlib snapshots settings
SNAP_MINUTES     = 5.0         # 5-minute snapshots
SNAP_ORIGIN_TS   = None        # None => align to floor(min(start_ts)/Δ)*Δ
SNAP_ID_BASE     = "input"     # {"input", "zero", "one"} for snapshot file node IDs
# --- NEW: ensure every window exists (even if there are no contacts in that window)
EMIT_EMPTY_WINDOWS = True      # fill gaps with a self-loop placeholder
# --- NEW: ensure every node is present in every window by adding a self-loop per node
#          This guarantees NDlib applies per-bin recovery (γ) to all infected nodes,
#          even when they have no contacts in that window.
PRESENCE_SELF_LOOPS_ALL_NODES = True
# ----------------------------------------------------

# Optional: import tacoma only if we need to write a .taco
try:
    import tacoma as tc
except Exception:
    tc = None

EPS = 1e-6  # seconds; tiny epsilon to separate simultaneous events and make tmax > last_t


def ts_to_iso_z(ts: int | float) -> str:
    """Convert UNIX timestamp to UTC ISO8601 'YYYY-MM-DDTHH:MM:SSZ'."""
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_preprocessed(path: str) -> pd.DataFrame:
    """Load preprocessed contacts (whitespace-delimited), columns: u v start_ts end_ts."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, :4]
    df.columns = ["u", "v", "start_ts", "end_ts"]

    # Ensure dtypes
    df["u"] = df["u"].astype(int)
    df["v"] = df["v"].astype(int)
    df["start_ts"] = df["start_ts"].astype(float)  # float to allow EPS operations
    df["end_ts"] = df["end_ts"].astype(float)

    # Drop zero/negative durations if any slipped through
    df = df[df["start_ts"] < df["end_ts"]].copy()
    return df


def format_replay(df: pd.DataFrame) -> str:
    """
    Produce EXACT format:
        NODE_LIST <sorted node ids>
        u, v, YYYY-MM-DDTHH:MM:SSZ, YYYY-MM-DDTHH:MM:SSZ;
        ...
    (Note the trailing semicolon and space at the end of each line.)
    """
    nodes = sorted(set(df["u"]).union(set(df["v"])))
    header = "NODE_LIST " + " ".join(map(str, nodes))

    df = df.sort_values(["u", "v", "start_ts", "end_ts"]).copy()
    df["start_iso"] = df["start_ts"].apply(ts_to_iso_z)
    df["end_iso"] = df["end_ts"].apply(ts_to_iso_z)

    lines = [f"{int(row.u)}, {int(row.v)}, {row.start_iso}, {row.end_iso}; " for _, row in df.iterrows()]
    return header + "\n" + "\n".join(lines)


def to_zero_based(df: pd.DataFrame, id_base: str) -> pd.DataFrame:
    """Return a copy with 0-based IDs if needed."""
    out = df.copy()
    if id_base.lower() in {"one", "1", "one-based", "one_based"}:
        out["u"] = out["u"].astype(int) - 1
        out["v"] = out["v"].astype(int) - 1
    else:
        out["u"] = out["u"].astype(int)
        out["v"] = out["v"].astype(int)
    return out

# --- NEW: Build a compact, sequential zero-based relabeling for node IDs
def relabel_sequential_zero_based(
    df_zero_based: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
    """
    Given a dataframe with 0-based IDs (possibly sparse, e.g., {0, 2, 7, ...}),
    return a copy where node IDs are remapped to 0..M-1 in ascending order
    of the original IDs. Also returns (old->new) and (new->old) maps.
    """
    nodes_sorted = sorted(set(df_zero_based["u"]).union(set(df_zero_based["v"])))
    id_map = {old: new for new, old in enumerate(nodes_sorted)}
    id_map_rev = {new: old for old, new in id_map.items()}

    out = df_zero_based.copy()
    out["u"] = out["u"].map(id_map).astype(int)
    out["v"] = out["v"].map(id_map).astype(int)
    return out, id_map, id_map_rev


def build_edge_changes(df_zero_based: pd.DataFrame):
    """
    Build a tacoma edge_changes object from (u, v, start_ts, end_ts) rows.

    - Normalizes endpoints so (u,v) == (v,u).
    - Creates an "in" event at start_ts and an "out" event at end_ts.
    - ε-extends any e<=s (shouldn't happen if preprocessed, but safe).
    - Sets N, t0, t, tmax, edges_in, edges_out, edges_initial, time_unit.
    """
    if tc is None:
        raise RuntimeError(
            "tacoma is not installed. Install with `pip install tacoma` to write a .taco file."
        )

    # Time-indexed event map
    evmap: Dict[float, Dict[str, List[Tuple[int, int]]]] = {}

    # Normalize undirected pairs and schedule events
    for u, v, s, e in df_zero_based[["u", "v", "start_ts", "end_ts"]].itertuples(index=False):
        uu, vv = (v, u) if u > v else (u, v)
        s = float(s)
        e = float(e)

        # Safety: ensure e > s (ε-extend if needed)
        if e <= s:
            e = s + EPS

        evmap.setdefault(s, {"in": [], "out": []})
        evmap.setdefault(e, {"in": [], "out": []})
        evmap[s]["in"].append((int(uu), int(vv)))
        evmap[e]["out"].append((int(uu), int(vv)))

    times = sorted(evmap.keys())
    nodes = sorted(set(df_zero_based["u"]).union(set(df_zero_based["v"])))
    N = (max(nodes) + 1) if nodes else 0

    # Derive t0/last_t from the actual event times
    first_t = float(times[0]) if times else 0.0
    last_t  = float(times[-1]) if times else 0.0
    t0   = first_t - EPS if times else 0.0   # ensure t0 < t[0]
    tmax = last_t  + EPS                     # ensure t[-1] < tmax
    tn = tc.edge_changes()
    tn.N = int(N)
    tn.t0 = float(t0)
    tn.tmax = float(tmax)
    tn.t = [float(t) for t in times]
    tn.edges_initial = []  # assume no edges already 'on' at t0
    tn.edges_in = [evmap[t]["in"] for t in times]
    tn.edges_out = [evmap[t]["out"] for t in times]
    tn.time_unit = "s"
    tn.notes = "Constructed from (u,v,start_ts,end_ts) intervals. Abutments merged upstream; ε used for safety."

    return tn


# -------------------- Snapshotting for NDlib/DyNetX --------------------

def _choose_origin(first_start_ts: float, window_s: float, user_origin_ts: Optional[float]) -> float:
    """
    Choose the epoch origin for snapshots.
    - If user provides SNAP_ORIGIN_TS, use it.
    - Else align to floor(first_start/window)*window so the first snapshot
      starts just before (or at) the first contact.
    """
    if user_origin_ts is not None:
        return float(user_origin_ts)
    if window_s <= 0:
        raise ValueError("Snapshot window must be > 0 seconds.")
    return math.floor(first_start_ts / window_s) * window_s


def generate_snapshot_rows(
    df: pd.DataFrame,
    window_s: float,
    origin_ts: Optional[float] = None,
    id_base: str = "input",
    emit_empty_windows: bool = False,
    # --- NEW: when True, add a (n,n,t) self-loop for every node n in every window t
    presence_all_nodes: bool = False
) -> List[Tuple[int, int, int]]:
    """
    Yield snapshot edgelist rows (u, v, t_idx) where t_idx is an integer snapshot id.

    Include (u, v, t) iff interval [s, e) overlaps snapshot window [T, T+Δ),
    with Δ = window_s and T = origin + t*Δ.

    id_base:
      - "input": keep node IDs as in df
      - "zero": force zero-based
      - "one":  force one-based

    NEW:
    If emit_empty_windows=True, ensure that every t_idx in the covered time span
    exists by inserting a single self-loop (i, i, t_idx) using the minimum node id
    seen in the data. This is a no-op for SIR dynamics but forces NDlib to advance
    time so recoveries proceed during contact-free periods.

    NEW:
    If presence_all_nodes=True, add a self-loop (n, n, t_idx) for *every* node n
    in *every* window t_idx within the covered time span. This guarantees that all
    nodes are processed each bin by NDlib, so recovery is applied even when nodes
    have no contacts in that window.
    """
    if df.empty:
        return []

    # ID normalization per requested base
    if id_base.lower() == "zero":
        df_use = to_zero_based(df, "one")  # treat input as one-based for shift by -1
    elif id_base.lower() == "one":
        df_use = df.copy()
        if df_use[["u", "v"]].min().min() == 0:
            df_use["u"] = df_use["u"].astype(int) + 1
            df_use["v"] = df_use["v"].astype(int) + 1
    else:
        df_use = df.copy()

    first_start = float(df_use["start_ts"].min())
    last_end    = float(df_use["end_ts"].max())
    origin = _choose_origin(first_start, window_s, origin_ts)

    # Snapshot index helpers
    def bin_start(t_idx: int) -> float:
        return origin + t_idx * window_s

    # We'll collect unique (t, u, v) so repeated overlaps in the same window de-dup
    seen = set()

    for u, v, s, e in df_use[["u", "v", "start_ts", "end_ts"]].itertuples(index=False):
        s = float(s)
        e = float(e)
        if e <= s:
            e = s + EPS

        # Compute generous index bounds; verify overlap explicitly.
        start_idx = int(math.floor((s - origin) / window_s) - 1)
        end_idx   = int(math.ceil((e - origin) / window_s) + 1)

        uu, vv = (v, u) if v < u else (u, v)
        for t_idx in range(start_idx, end_idx + 1):
            T = bin_start(t_idx)
            if (s < T + window_s) and (e > T):  # overlap
                seen.add((t_idx, int(uu), int(vv)))

    # --- NEW: compute node universe once (in the chosen id_base)
    nodes_set = set(df_use["u"]).union(set(df_use["v"]))

    # --- NEW: force existence of every window in the covered time span
    if emit_empty_windows:
        sentinel  = int(min(nodes_set)) if nodes_set else 0  # real node, stable across all windows
        # windows covering [first_start, last_end)
        first_t_idx = int(math.floor((first_start - origin) / window_s))          # usually 0
        last_t_idx_excl = int(math.ceil((last_end - origin) / window_s))          # exclusive
        existing_t = {t for (t, _, _) in seen}
        for t_idx in range(first_t_idx, last_t_idx_excl):
            if t_idx not in existing_t:
                seen.add((t_idx, sentinel, sentinel))  # self-loop placeholder

    # --- NEW: ensure *every node* is present in *every window*
    if presence_all_nodes and nodes_set:
        first_t_idx = int(math.floor((first_start - origin) / window_s))
        last_t_idx_excl = int(math.ceil((last_end - origin) / window_s))
        for t_idx in range(first_t_idx, last_t_idx_excl):
            for n in nodes_set:
                seen.add((t_idx, int(n), int(n)))  # self-loop per node

    # Sort by t_idx, then u, v for stable output
    rows = [(u, v, t_idx) for (t_idx, u, v) in sorted(seen)]
    return rows


def write_ndlib_snapshots_file(
    rows: Iterable[Tuple[int, int, int]],
    path: str
) -> None:
    """Write snapshot rows as 'u v t' (each on its own line)."""
    with open(path, "w", encoding="utf-8") as f:
        for u, v, t in rows:
            f.write(f"{int(u)} {int(v)} {int(t)}\n")

def write_epihiper_exports(df: pd.DataFrame, out_dir: str, time_unit: str = "s") -> None:
    """
    Emits:
      - nodes.csv               (node_id)
      - contacts_intervals.csv  (u,v,start_ts,end_ts)
      - contacts_events.csv     (t,op,u,v) with op in {+1,-1}
      - metadata.json           {N,t0,tmax,time_unit}
    Assumes df has 0-based, sequential IDs (use your existing relabel).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Universe of nodes
    nodes = sorted(set(df["u"]).union(df["v"]))
    with open(os.path.join(out_dir, "nodes.csv"), "w", encoding="utf-8") as f:
        f.write("node_id\n")
        for n in nodes:
            f.write(f"{int(n)}\n")

    # Interval contacts
    with open(os.path.join(out_dir, "contacts_intervals.csv"), "w", encoding="utf-8") as f:
        f.write("u,v,start_ts,end_ts\n")
        for u, v, s, e in df[["u", "v", "start_ts", "end_ts"]].itertuples(index=False):
            if e <= s:
                e = float(s) + EPS
            uu, vv = (v, u) if u > v else (u, v)  # normalize undirected
            f.write(f"{int(uu)},{int(vv)},{int(s)},{int(e)}\n")

    # Eventized contacts (+1 at start, -1 at end)
    events = []
    for u, v, s, e in df[["u", "v", "start_ts", "end_ts"]].itertuples(index=False):
        if e <= s:
            e = float(s) + EPS
        uu, vv = (v, u) if u > v else (u, v)
        events.append((float(s), +1, int(uu), int(vv)))
        events.append((float(e), -1, int(uu), int(vv)))
    events.sort(key=lambda r: (r[0], -r[1], r[2], r[3]))  # start(+1) before end(-1) at same t

    with open(os.path.join(out_dir, "contacts_events.csv"), "w", encoding="utf-8") as f:
        f.write("t,op,u,v\n")
        for t, op, uu, vv in events:
            f.write(f"{int(t)},{op},{uu},{vv}\n")

    # Metadata
    t_values = [float(x) for x in df["start_ts"]] + [float(x) for x in df["end_ts"]]
    if t_values:
        t0 = min(t_values) - EPS
        tmax = max(t_values) + EPS
    else:
        t0 = 0.0
        tmax = 0.0
    meta = {
        "N": int(max(nodes) + 1 if nodes else 0),
        "t0": float(t0),
        "tmax": float(tmax),
        "time_unit": time_unit,
        "notes": "Generated from interval contacts; undirected normalized; EPS used for abutments."
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# -------------------- Minimal main --------------------

def main():
    """Minimal runner: reads INPUT_PATH and writes replay, .taco, and NDLib snapshots."""
    df = load_preprocessed(INPUT_PATH)

    if WRITE_REPLAY:
        out_replay = format_replay(df)
        with open(OUT_TEXT_PATH, "w", encoding="utf-8") as f:
            f.write(out_replay)

    if WRITE_TACO:
        if tc is None:
            raise RuntimeError(
                "tacoma is not installed. Install with `pip install tacoma` or set WRITE_TACO = False."
            )
        df0 = to_zero_based(df, ID_BASE_FOR_TACOMA)

        # --- NEW: sequentially remap IDs ONLY for the .taco output
        if SEQUENTIALIZE_TACO_NODE_IDS:
            df_seq, id_map, id_map_rev = relabel_sequential_zero_based(df0)
        else:
            df_seq = df0  # no remap

        tn = build_edge_changes(df_seq)
        tc.verify(tn)

        # --- NEW: annotate the .taco with a note about remapping
        if SEQUENTIALIZE_TACO_NODE_IDS:
            orig_unique = len(set(df0["u"]).union(set(df0["v"])))
            note = f" IDs remapped to sequential zero-based for .taco output ({orig_unique} unique -> {tn.N})."
            tn.notes = (tn.notes + note) if getattr(tn, "notes", "") else note

        with open(OUT_TACO_PATH, "w", encoding="utf-8") as f:
            tc.write_json_taco(tn, f)

    if WRITE_EPIHIPER:
        df_for_epi = df0.copy()
        if SEQUENTIALIZE_TACO_NODE_IDS:
            df_for_epi, _, _ = relabel_sequential_zero_based(df_for_epi)
        write_epihiper_exports(df_for_epi, EPIHIPER_DIR, EPIHIPER_TIME_UNIT)
        print(f"EpiHIPER export written to '{EPIHIPER_DIR}/' "
              f"(nodes.csv, contacts_intervals.csv, contacts_events.csv, metadata.json)")

    if WRITE_NDLIB_SNAPSHOTS:
        window_s = float(SNAP_MINUTES) * 60.0
        rows = generate_snapshot_rows(
            df=df,
            window_s=window_s,
            origin_ts=SNAP_ORIGIN_TS,
            id_base=SNAP_ID_BASE,
            emit_empty_windows=EMIT_EMPTY_WINDOWS,
            # --- NEW: make sure every node appears each window
            presence_all_nodes=PRESENCE_SELF_LOOPS_ALL_NODES
        )
        write_ndlib_snapshots_file(rows, SNAPSHOTS_OUT)

        # Tiny log for sanity
        if len(rows) > 0:
            first_start = float(df["start_ts"].min())
            effective_origin = SNAP_ORIGIN_TS if SNAP_ORIGIN_TS is not None else (
                math.floor(first_start / window_s) * window_s
            )
            print(
                f"Wrote {len(rows)} snapshot edges to '{SNAPSHOTS_OUT}'. "
                f"Δ={int(window_s)}s; origin={int(effective_origin)} "
                f"(first window [{int(effective_origin)}, {int(effective_origin + window_s)}))"
            )
        else:
            print(f"No snapshot edges emitted to '{SNAPSHOTS_OUT}' (empty input).")


if __name__ == "__main__":
    main()
