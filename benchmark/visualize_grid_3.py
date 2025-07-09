import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# Configure PGF export for LaTeX
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "pgf.rcfonts": False,
})

# Constants
OUTPUT_DIR = "plots"
OUTPUT_NAME = "runtime_block_plot"
LARGE_FONT = 12

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv("3d_runtime_surface.csv")
valid = df["gpu_runtime_seconds"].notna()

# Normalize edge counts for display (millions, not billions)
df["edges_million"] = df["num_edges"] / 1e6
x_m_unique = np.sort(df.loc[valid, "edges_million"].unique())
y_unique = np.sort(df.loc[valid, "num_parallel_simulations"].unique())

# Create grid for GPU runtimes
Z_grid = np.full((len(y_unique), len(x_m_unique)), np.nan)
for i, y_val in enumerate(y_unique):
    for j, x_val in enumerate(x_m_unique):
        match = df[(np.isclose(df["edges_million"], x_val)) &
                   (df["num_parallel_simulations"] == y_val)]
        if not match["gpu_runtime_seconds"].isna().all():
            Z_grid[i, j] = match["gpu_runtime_seconds"].values[0]

# Compute bin edges
x_step = x_m_unique[1] - x_m_unique[0] if len(x_m_unique) > 1 else 1.0
y_step = y_unique[1] - y_unique[0] if len(y_unique) > 1 else 100_000
x_edges = np.concatenate(([x_m_unique[0] - x_step / 2],
                          (x_m_unique[:-1] + x_m_unique[1:]) / 2,
                          [x_m_unique[-1] + x_step / 2]))
y_edges = np.concatenate(([y_unique[0] - y_step / 2],
                          (y_unique[:-1] + y_unique[1:]) / 2,
                          [y_unique[-1] + y_step / 2]))

# Plot
plt.figure(figsize=(10, 7))
ax = plt.gca()
c = ax.pcolormesh(x_edges, y_edges, Z_grid, cmap='viridis', shading='auto')

# Colorbar
cbar = plt.colorbar(c, ax=ax)
cbar.set_label("Total Runtime for 100 Simulation Steps (s)", fontsize=LARGE_FONT)
cbar.ax.tick_params(labelsize=LARGE_FONT)

# X-axis: ticks and labels (in millions)
x_ticks_forced = np.arange(5, 55, 5)
ax.set_xticks(x_ticks_forced)
ax.set_xticklabels([f"{v:.0f}" for v in x_ticks_forced], rotation=0, ha='center', fontsize=LARGE_FONT)

# Y-axis: format in K/M
ax.set_yticks(y_unique)
ax.set_yticklabels([f"{int(v/1e6)}M" if v >= 1e6 else f"{int(v/1e3)}K" for v in y_unique], fontsize=LARGE_FONT)

# Axis labels and title
ax.set_title("Simulation Runtime - 1000 Individuals", fontsize=LARGE_FONT)
ax.set_xlabel("Simulated Contacts (Millions)", fontsize=LARGE_FONT)
ax.set_ylabel("Parallel Simulations", fontsize=LARGE_FONT)

# Ticks padding
ax.tick_params(axis='x', pad=5)
ax.tick_params(axis='y', pad=5)

# Grid and layout
ax.grid(False)
plt.tight_layout()

# Save in all formats
for ext in ["png", "pdf", "pgf"]:
    plt.savefig(os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}.{ext}"), bbox_inches='tight')

plt.close()
