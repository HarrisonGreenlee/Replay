#!/usr/bin/env python3
"""
contactgraph.py

Reads a plain-text command file (one command per line) and writes a contact-graph
file with:

  - A single header line:
        NODE_LIST <id1> <id2> ...
    with strictly ascending, de-duplicated node IDs.

  - One line per edge occurrence:
        u, v, YYYY-MM-DDTHH:MM:SSZ, YYYY-MM-DDTHH:MM:SSZ;
    EXACT spacing:
      * exactly one space after each comma
      * ISO8601 UTC timestamps with trailing 'Z'
      * each line ends with ';' followed by exactly one space, then newline

Commands (per-line):
  ADDNODE <ID>
  ADDEDGE [ID1,ID2,ID3,...] <UNIX_START> <DURATION> <RECURRENCE_DELTA> [OCCURRENCES N | UNTIL <ISO8601Z|UNIX>]

Notes:
  - IDs must be positive integers.
  - UNIX times and durations are unsigned integers in seconds.
  - For ADDEDGE, every unordered pair among the listed IDs (complete graph) is emitted
    for each occurrence time t_k = START + k * DELTA.
  - Horizon is PER EDGE:
      * OCCURRENCES N → k = 0..N-1
      * UNTIL T      → all k with t_k < T (T can be ISO8601Z or UNIX seconds)
    If neither is provided, one occurrence (k = 0) is generated.
    If RECURRENCE_DELTA is 0, only k = 0 is generated REGARDLESS of horizon.
  - If any ID in an ADDEDGE is undefined, emit an error and skip the entire ADDEDGE line.
  - If both OCCURRENCES and UNTIL are present, emit an error and skip the line.
  - ID lists with fewer than two unique IDs are skipped with a warning.
  - Identical occurrences (u, v, start, end) are emitted only once.
  - Output occurrences are sorted by (start, u, v, end) for determinism.

Usage:
  python contactgraph.py <commands.txt> --out <graph.txt>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import itertools
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple


# ---------- Utilities ----------

def _err(msg: str) -> None:
    """Print an error to stderr."""
    sys.stderr.write(msg.rstrip() + "\n")


def _warn(msg: str) -> None:
    """Print a warning to stderr."""
    sys.stderr.write(msg.rstrip() + "\n")


def _iso8601z_from_unix(ts: int) -> str:
    """Convert UNIX seconds to ISO8601 UTC string with trailing 'Z'."""
    # Python guarantees seconds precision here; ensure exact 'Z' suffix.
    return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_unsigned_int(token: str, *, context: str, line_no: int) -> Optional[int]:
    """Parse an unsigned integer token; on failure, emit error and return None."""
    if not token.isdigit():
        _err(f"ERROR: Invalid unsigned integer '{token}' for {context} on line {line_no}. Skipping line {line_no}.")
        return None
    try:
        val = int(token, 10)
    except Exception:
        _err(f"ERROR: Could not parse integer '{token}' for {context} on line {line_no}. Skipping line {line_no}.")
        return None
    if val < 0:
        _err(f"ERROR: Negative value '{token}' for {context} on line {line_no}. Skipping line {line_no}.")
        return None
    return val


def _parse_iso8601z(s: str, *, line_no: int) -> Optional[int]:
    """
    Parse an ISO8601 UTC timestamp of the exact form YYYY-MM-DDTHH:MM:SSZ into UNIX seconds.
    Returns None on failure (with error already emitted).
    """
    try:
        dt = _dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
        dt = dt.replace(tzinfo=_dt.timezone.utc)
        return int(dt.timestamp())
    except Exception:
        _err(f"ERROR: Invalid ISO8601Z time '{s}' on line {line_no}. Expected 'YYYY-MM-DDTHH:MM:SSZ'.")
        return None


def _parse_until_token(tok: str, *, line_no: int) -> Optional[int]:
    """
    Parse UNTIL token which can be:
      - UNIX seconds (unsigned integer string), or
      - ISO8601Z (YYYY-MM-DDTHH:MM:SSZ)
    Returns UNIX seconds on success; None on failure (with error emitted).
    """
    if tok.isdigit():
        val = int(tok, 10)
        if val < 0:
            _err(f"ERROR: UNTIL negative UNIX time '{tok}' on line {line_no}.")
            return None
        return val
    return _parse_iso8601z(tok, line_no=line_no)


def _find_matching_bracket(s: str, start_idx: int) -> int:
    """Find the index of the matching ']' for a '[' at start_idx. Returns -1 if not found."""
    depth = 0
    for i in range(start_idx, len(s)):
        c = s[i]
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                return i
    return -1


def _parse_id_list(src: str, start_idx: int, *, line_no: int) -> Tuple[Optional[List[int]], int]:
    """
    Parse an ID list token starting at src[start_idx] which must be '['.
    Returns (list_of_ids, next_index_after_list). On error, (None, start_idx).
    """
    if start_idx >= len(src) or src[start_idx] != '[':
        _err(f"ERROR: Expected '[' to start ID list on line {line_no}. Skipping line {line_no}.")
        return None, start_idx
    end_idx = _find_matching_bracket(src, start_idx)
    if end_idx == -1:
        _err(f"ERROR: Unterminated ID list '[' on line {line_no}. Skipping line {line_no}.")
        return None, start_idx

    inner = src[start_idx + 1:end_idx]
    # Split by comma; allow arbitrary surrounding whitespace; reject empty items (except totally empty list).
    if inner.strip() == "":
        ids: List[int] = []
    else:
        parts = [p.strip() for p in inner.split(',')]
        ids = []
        for p in parts:
            if p == "":
                _err(f"ERROR: Empty entry in ID list on line {line_no}. Skipping line {line_no}.")
                return None, start_idx
            if not p.isdigit():
                _err(f"ERROR: Non-integer ID '{p}' on line {line_no}. Skipping line {line_no}.")
                return None, start_idx
            val = int(p, 10)
            if val <= 0:
                _err(f"ERROR: Node IDs must be positive integers. Found '{p}' on line {line_no}. Skipping line {line_no}.")
                return None, start_idx
            ids.append(val)
    return ids, end_idx + 1


@dataclass(frozen=True, order=False)
class EdgeOccurrence:
    """A single contact between two nodes from start_ts to end_ts (UNIX seconds)."""
    u: int
    v: int
    start_ts: int
    end_ts: int

    def sort_key(self) -> Tuple[int, int, int, int]:
        return (self.start_ts, self.u, self.v, self.end_ts)


# ---------- Core processing ----------

def _generate_pairs(ids: Sequence[int]) -> Iterable[Tuple[int, int]]:
    """Yield unordered pairs (u, v) with u < v from a sorted sequence of unique IDs."""
    for u, v in itertools.combinations(ids, 2):
        if u < v:
            yield (u, v)
        else:
            yield (v, u)


def _process_addedge_line(
    line: str,
    line_no: int,
    known_nodes: Set[int],
    occurrences_out: Set[EdgeOccurrence],
) -> None:
    """
    Parse and process an ADDEDGE line. On errors, emit messages and skip this line.
    Adds generated EdgeOccurrence objects to occurrences_out.
    """
    # Remove leading "ADDEDGE" and proceed.
    remainder = line[len("ADDEDGE"):].strip()

    # 1) Parse ID list
    if not remainder or remainder[0] != '[':
        _err(f"ERROR: Expected ID list in brackets after ADDEDGE on line {line_no}. Skipping ADDEDGE on line {line_no}.")
        return
    id_list, idx_after_list = _parse_id_list(remainder, 0, line_no=line_no)
    if id_list is None:
        return  # Error already emitted
    # Collapse to sorted unique IDs
    unique_ids = sorted(set(id_list))
    if len(unique_ids) < 2:
        _warn(f"WARNING: ADDEDGE on line {line_no} has fewer than two unique IDs; skipping.")
        return

    # 2) Parse three unsigned integers: start, duration, delta
    rest = remainder[idx_after_list:].strip()
    if not rest:
        _err(f"ERROR: Missing timing parameters after ID list on line {line_no}. Skipping ADDEDGE on line {line_no}.")
        return
    # Tokenize by whitespace
    tokens = rest.split()
    if len(tokens) < 3:
        _err(f"ERROR: Expected <UNIX_START> <DURATION> <RECURRENCE_DELTA> on line {line_no}. Skipping ADDEDGE on line {line_no}.")
        return

    start_unix = _parse_unsigned_int(tokens[0], context="UNIX_START", line_no=line_no)
    duration = _parse_unsigned_int(tokens[1], context="DURATION", line_no=line_no)
    delta = _parse_unsigned_int(tokens[2], context="RECURRENCE_DELTA", line_no=line_no)
    if start_unix is None or duration is None or delta is None:
        _err(f"Skipping ADDEDGE on line {line_no}.")
        return

    # 3) Optional horizon: OCCURRENCES N OR UNTIL T (but not both)
    horizon_kind: Optional[str] = None  # "OCCURRENCES" or "UNTIL" or None
    occurrences_count: Optional[int] = None
    until_unix: Optional[int] = None

    extra_tokens = tokens[3:]
    if extra_tokens:
        # Detect conflicting horizon keywords anywhere in the remainder
        has_occ = "OCCURRENCES" in extra_tokens
        has_until = "UNTIL" in extra_tokens
        if has_occ and has_until:
            _err(f"ERROR: Both OCCURRENCES and UNTIL specified on line {line_no}. Skipping ADDEDGE on line {line_no}.")
            return

        # Parse the optional clause precisely
        # Reconstruct the remainder as a space-separated string to parse predictable positions.
        # Expected forms:
        #   OCCURRENCES N
        #   UNTIL T
        if extra_tokens[0] == "OCCURRENCES":
            horizon_kind = "OCCURRENCES"
            if len(extra_tokens) != 2:
                _err(f"ERROR: 'OCCURRENCES' must be followed by a single unsigned integer on line {line_no}. Skipping ADDEDGE on line {line_no}.")
                return
            n = _parse_unsigned_int(extra_tokens[1], context="OCCURRENCES", line_no=line_no)
            if n is None:
                _err(f"Skipping ADDEDGE on line {line_no}.")
                return
            occurrences_count = n
        elif extra_tokens[0] == "UNTIL":
            horizon_kind = "UNTIL"
            if len(extra_tokens) != 2:
                _err(f"ERROR: 'UNTIL' must be followed by exactly one time token on line {line_no}. Skipping ADDEDGE on line {line_no}.")
                return
            u = _parse_until_token(extra_tokens[1], line_no=line_no)
            if u is None:
                _err(f"Skipping ADDEDGE on line {line_no}.")
                return
            until_unix = u
        else:
            # Unrecognized trailing tokens
            _err(f"ERROR: Unexpected tokens after timing parameters on line {line_no}: '{' '.join(extra_tokens)}'. Skipping ADDEDGE on line {line_no}.")
            return

    # 4) Verify all IDs are defined as nodes at this point
    undefined = [x for x in unique_ids if x not in known_nodes]
    if undefined:
        # Emit one error per undefined ID, but one skip line total (as per guide's spirit)
        for x in undefined:
            _err(f"ERROR: ID {x} is undefined. Add it with `ADDNODE {x}`.")
        _err(f"Skipping ADDEDGE on line {line_no}.")
        return

    # 5) Generate occurrences
    #    If delta == 0 → exactly one occurrence (k = 0) REGARDLESS of horizon
    starts: List[int] = []
    if delta == 0:
        starts = [start_unix]
    else:
        if horizon_kind == "OCCURRENCES":
            n = occurrences_count if occurrences_count is not None else 1
            # If N == 0, nothing is emitted
            for k in range(n):
                starts.append(start_unix + k * delta)
        elif horizon_kind == "UNTIL":
            T = until_unix if until_unix is not None else start_unix
            # Generate t_k < T
            t = start_unix
            while t < T:
                starts.append(t)
                t += delta
        else:
            # Default: single occurrence
            starts = [start_unix]

    # 6) Add occurrences for all unordered pairs (u < v)
    sorted_ids = unique_ids  # already sorted
    for t0 in starts:
        t1 = t0 + duration
        for (u, v) in _generate_pairs(sorted_ids):
            occurrences_out.add(EdgeOccurrence(u=u, v=v, start_ts=t0, end_ts=t1))


def process_command_file(cmd_path: str) -> Tuple[List[int], List[EdgeOccurrence]]:
    """
    Read and process the command file.

    Returns:
        (sorted_nodes, sorted_edge_occurrences)
    """
    nodes: Set[int] = set()
    occs: Set[EdgeOccurrence] = set()

    with open(cmd_path, "r", encoding="utf-8") as f:
        for idx, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue  # skip empty
            if line.startswith("#"):
                continue  # comment

            if line.startswith("ADDNODE"):
                rest = line[len("ADDNODE"):].strip()
                if rest == "":
                    _err(f"ERROR: Missing ID for ADDNODE on line {idx}. Skipping line {idx}.")
                    continue
                # Only one token allowed
                parts = rest.split()
                if len(parts) != 1:
                    _err(f"ERROR: ADDNODE expects exactly one ID on line {idx}. Skipping line {idx}.")
                    continue
                node_id = _parse_unsigned_int(parts[0], context="node ID", line_no=idx)
                if node_id is None:
                    _err(f"Skipping ADDNODE on line {idx}.")
                    continue
                if node_id <= 0:
                    _err(f"ERROR: Node IDs must be positive integers. Found '{node_id}' on line {idx}. Skipping line {idx}.")
                    continue
                nodes.add(node_id)

            elif line.startswith("ADDEDGE"):
                _process_addedge_line(line, idx, nodes, occs)

            else:
                _err(f"ERROR: Unknown command on line {idx}: '{line}'. Skipping line {idx}.")

    # Prepare deterministic outputs
    node_list_sorted = sorted(nodes)
    occurrences_sorted = sorted(occs, key=lambda e: e.sort_key())
    return node_list_sorted, occurrences_sorted


def write_contact_graph(out_path: str, nodes_sorted: Sequence[int], occs_sorted: Sequence[EdgeOccurrence]) -> None:
    """
    Write the contact graph to `out_path` with the exact formatting requirements.
    """
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        if nodes_sorted:
            out.write("NODE_LIST " + " ".join(str(n) for n in nodes_sorted) + "\n")
        else:
            # No trailing space when no IDs are present
            out.write("NODE_LIST\n")

        for e in occs_sorted:
            start_iso = _iso8601z_from_unix(e.start_ts)
            end_iso = _iso8601z_from_unix(e.end_ts)
            # EXACT spacing: "u, v, <start>, <end>; " then newline
            out.write(f"{e.u}, {e.v}, {start_iso}, {end_iso}; \n")


# ---------- CLI ----------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a contact-graph file from a command file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("commands", help="Path to the command file (text).")
    p.add_argument("--out", required=True, help="Path to write the contact graph (text).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = _build_argparser()
    args = ap.parse_args(argv)

    try:
        nodes, occs = process_command_file(args.commands)
    except FileNotFoundError:
        _err(f"ERROR: Command file not found: {args.commands}")
        return 1
    except PermissionError:
        _err(f"ERROR: Permission denied when reading: {args.commands}")
        return 1

    try:
        write_contact_graph(args.out, nodes, occs)
    except PermissionError:
        _err(f"ERROR: Permission denied when writing: {args.out}")
        return 1
    except OSError as e:
        _err(f"ERROR: Failed to write output '{args.out}': {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
