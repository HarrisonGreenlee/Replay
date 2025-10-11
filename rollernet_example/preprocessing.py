# preprocess_contacts.py
# - Reads contacts from a whitespace-delimited text file (>= 4 cols: u v start_ts end_ts ...)
# - Normalizes undirected edges (u <= v)
# - Merges overlapping **and abutting** intervals per edge
# - Drops zero/negative-duration intervals (start_ts < end_ts only)
# - Expands the dataset with +5d and +7d offsets, repeated for a given number of weeks
# - Optionally omits the very last Sunday so the range is exactly N weeks (no extra day)
# - Saves back in the SAME format as original: whitespace-delimited "u v start_ts end_ts" (no header)

from __future__ import annotations
import argparse
from typing import List
import pandas as pd
from datetime import datetime, UTC

# -------- Defaults (overridable via CLI) --------
DEFAULT_INPUT = "imote-traces-RollerNet/contacts.dat"
DEFAULT_OUTPUT = "contacts_preprocessed.dat"
DEFAULT_WEEKS = 4
DEFAULT_OMIT_LAST_DAY = True
# ------------------------------------------------


def load_contacts(filepath: str) -> pd.DataFrame:
    """Load contact data (whitespace-delimited). Uses the first 4 columns: u v start_ts end_ts.
    Drops zero/negative-duration intervals proactively.
    """
    df = pd.read_csv(filepath, sep=r"\s+", header=None).iloc[:, :4]
    df.columns = ["u", "v", "start_ts", "end_ts"]

    # Keep strictly positive durations only
    df = df[df["start_ts"] < df["end_ts"]].copy()

    # Ensure integer-like columns are ints
    df["u"] = df["u"].astype(int)
    df["v"] = df["v"].astype(int)
    df["start_ts"] = df["start_ts"].astype(int)
    df["end_ts"] = df["end_ts"].astype(int)
    return df


def normalize_undirected(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize edges so (u,v) == (v,u) by enforcing u<=v."""
    uv = df.apply(
        lambda r: pd.Series({"u": min(r.u, r.v), "v": max(r.u, r.v)}),
        axis=1,
    )
    out = pd.concat([uv, df[["start_ts", "end_ts"]]], axis=1)
    return out


def _merge_overlaps_and_abutments_for_edge(g: pd.DataFrame) -> pd.DataFrame:
    """
    Coalesce intervals for one undirected edge group (u,v):
    - Sort by start_ts, end_ts
    - Merge if next.start_ts <= cur.end_ts (overlap OR abutment)
      (treat intervals as continuous when touching at the boundary)
    """
    g = g.sort_values(by=["start_ts", "end_ts"]).reset_index(drop=True)
    merged: list[dict] = []
    for _, row in g.iterrows():
        if not merged:
            merged.append(row.to_dict())
            continue

        cur = merged[-1]
        # Merge overlap OR abutment
        if row["start_ts"] <= cur["end_ts"]:
            cur["end_ts"] = max(cur["end_ts"], row["end_ts"])
        else:
            merged.append(row.to_dict())

    return pd.DataFrame(merged)


def dedup_symmetric_and_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize undirected edges and coalesce intervals (overlap + abutment)."""
    undirected = normalize_undirected(df)
    deduped = (
        undirected.groupby(["u", "v"], as_index=False, group_keys=False)
        .apply(_merge_overlaps_and_abutments_for_edge)
        .reset_index(drop=True)
    )
    return deduped


def build_offsets(weeks: int, omit_last_day: bool) -> List[int]:
    """
    Build second-based offsets for a Fri/Sun cadence over `weeks`.
    Pattern per week w (starting at 0): +5d and +7d added to w*7d baseline.
    Includes the original day (0 seconds) once at the beginning.

    Example weeks=4 (in days): [0, 5, 7, 12, 14, 19, 21, 26, 28]
    If omit_last_day: remove final 28 -> [0, 5, 7, 12, 14, 19, 21, 26]
    """
    d5 = 5 * 86400
    d7 = 7 * 86400

    offsets = [0]
    for w in range(weeks):
        base = w * d7
        offsets.append(base + d5)  # Friday
        offsets.append(base + d7)  # Sunday

    if omit_last_day and len(offsets) > 1:
        offsets.pop()

    return offsets


def apply_offsets(deduped: pd.DataFrame, offsets_sec: List[int]) -> pd.DataFrame:
    """Duplicate rows for each offset and shift timestamps by that many seconds."""
    parts = []
    for off in offsets_sec:
        block = deduped.copy()
        block["start_ts"] = block["start_ts"] + off
        block["end_ts"] = block["end_ts"] + off
        parts.append(block)
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["u", "v", "start_ts", "end_ts"]).reset_index(drop=True)
    return out


def save_as_original_format(df: pd.DataFrame, path: str) -> None:
    """Save as whitespace-delimited 'u v start_ts end_ts' with NO header, matching the original style."""
    df[["u", "v", "start_ts", "end_ts"]].to_csv(path, sep=" ", header=False, index=False)


def main():
    parser = argparse.ArgumentParser(description="Preprocess contact data.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to input contacts (whitespace-delimited).")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Where to save the preprocessed contacts.")
    parser.add_argument("--weeks", type=int, default=DEFAULT_WEEKS, help="Number of recurrence weeks to add (Fri/Sun).")
    parser.add_argument(
        "--omit-last-day",
        action="store_true",
        default=DEFAULT_OMIT_LAST_DAY,
        help="Omit the final Sunday (avoid Nw + 1 day).",
    )
    parser.add_argument(
        "--keep-last-day",
        action="store_true",
        help="Override to KEEP the final Sunday.",
    )
    args = parser.parse_args()

    omit_last_day = args.omit_last_day
    if args.keep_last_day:
        omit_last_day = False

    base = load_contacts(args.input)
    base_dedup = dedup_symmetric_and_overlaps(base)

    offsets = build_offsets(args.weeks, omit_last_day)
    expanded = apply_offsets(base_dedup, offsets)

    save_as_original_format(expanded, args.output)

    earliest_start = expanded["start_ts"].min()
    latest_end = expanded["end_ts"].max()

    earliest_iso = datetime.fromtimestamp(earliest_start, UTC).isoformat()
    latest_iso = datetime.fromtimestamp(latest_end, UTC).isoformat()

    print(f"Earliest timestamp: {earliest_start:<12} ({earliest_iso})")
    print(f"Latest timestamp:   {latest_end:<12} ({latest_iso})")


if __name__ == "__main__":
    main()
