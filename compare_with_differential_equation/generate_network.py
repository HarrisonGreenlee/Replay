import random
from datetime import datetime, timedelta

# Configuration
random.seed(42)
num_nodes = 1000
# num_groups = 10
num_groups = 1
group_size = num_nodes // num_groups
inter_group_prob = 0.01
num_days = 100
start_date = datetime.strptime("2000-01-01T00:00:00", "%Y-%m-%dT%H:%M:%S")
output_file = "synthetic_temporal_graph.txt"

# Group assignment
group_assignments = {i: i // group_size for i in range(num_nodes)}

# Collect edges in a list
edges = []

for day in range(num_days):
    day_start = (start_date + timedelta(days=day)).strftime("%Y-%m-%dT%H:%M:%SZ")
    day_end = (start_date + timedelta(days=day + 1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Intragroup edges (fully connected)
    for g in range(num_groups):
        nodes = [i for i in range(num_nodes) if group_assignments[i] == g]
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                edges.append((nodes[i] + 1, nodes[j] + 1, day_start, day_end))
    
    # Intergroup edges (sparse)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if group_assignments[i] != group_assignments[j]:
                if random.random() < inter_group_prob:
                    edges.append((i + 1, j + 1, day_start, day_end))

# Write to file in specified format
with open(output_file, "w") as f:
    # Write node list
    f.write("NODE_LIST")
    for node_id in range(1, num_nodes + 1):
        f.write(f" {node_id}")
    f.write("\n")
    
    # Write number of edges
    # no longer used for this file format
    # f.write(f"NUM_EDGES {len(edges)}\n")
    
    # Write each edge
    for src, tgt, start_ts, end_ts in edges:
        f.write(f"{src}, {tgt}, {start_ts}, {end_ts};\n")
