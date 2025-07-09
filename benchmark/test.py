from collections import defaultdict
import re

INPUT_FILE = "dense_1000.txt" 
MAX_EDGES = 10_000_000

unique_pairs = set()
line_count = 0
pair_counts = defaultdict(int)

with open(INPUT_FILE, "r") as f:
    for line in f:
        if line.startswith("EDGE_LIST") or line.startswith("NUM_EDGES"):
            continue

        # Parse line like: 1, 2, 2000-01-01T00:00:00Z, 2000-01-01T00:01:00Z;
        match = re.match(r'\s*(\d+),\s*(\d+),', line)
        if not match:
            continue

        a, b = int(match.group(1)), int(match.group(2))
        node_pair = tuple(sorted((a, b)))
        unique_pairs.add(node_pair)
        pair_counts[node_pair] += 1

        line_count += 1
        if line_count % 1_000_000 == 0:
            print(f"[Progress] Parsed {line_count} edges...")

        if line_count >= MAX_EDGES:
            break

print(f"\n[RESULTS]")
print(f"Total parsed edges: {line_count}")
print(f"Unique undirected node pairs: {len(unique_pairs)}")
print(f"Sample node pairs:")
for i, pair in enumerate(list(unique_pairs)[:10]):
    print(f"  {pair} — {pair_counts[pair]} edges")
