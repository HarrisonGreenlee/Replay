from __future__ import annotations
from datetime import datetime, timezone
import pandas as pd

INPUT_PATH    = "contacts_preprocessed.dat"
OUT_TEXT_PATH = "contacts_replay.txt"

def ts_to_iso_z(ts: int | float) -> str:
    """Convert UNIX timestamp to UTC ISO8601 'YYYY-MM-DDTHH:MM:SSZ'."""
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def load_preprocessed(path: str) -> pd.DataFrame:
    """Load preprocessed contacts (whitespace-delimited), columns: u v start_ts end_ts."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["u", "v", "start_ts", "end_ts"])
    df = df[df["start_ts"] < df["end_ts"]]  # Drop invalid intervals
    return df

def format_replay(df: pd.DataFrame) -> str:
    """
    Produce EXACT format:
        NODE_LIST <sorted node ids>
        u, v, YYYY-MM-DDTHH:MM:SSZ, YYYY-MM-DDTHH:MM:SSZ;
    """
    nodes = sorted(set(df["u"]).union(df["v"]))
    header = "NODE_LIST " + " ".join(map(str, nodes))
    df = df.sort_values(["u", "v", "start_ts", "end_ts"])
    df["start_iso"] = df["start_ts"].apply(ts_to_iso_z)
    df["end_iso"] = df["end_ts"].apply(ts_to_iso_z)
    lines = [f"{int(r.u)}, {int(r.v)}, {r.start_iso}, {r.end_iso}; " for _, r in df.iterrows()]
    return header + "\n" + "\n".join(lines)

def main():
    df = load_preprocessed(INPUT_PATH)
    out = format_replay(df)
    with open(OUT_TEXT_PATH, "w", encoding="utf-8") as f:
        f.write(out)

if __name__ == "__main__":
    main()