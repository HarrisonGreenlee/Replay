import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

# Load the CSV
csv_path = "3d_runtime_surface.csv"
df = pd.read_csv(csv_path)

# Remove rows with NaN (failed simulations)
df = df.dropna()

# Extract axes
X = df["num_edges"]
Y = df["num_parallel_simulations"]
CPU_Z = df["cpu_runtime_seconds"]
GPU_Z = df["gpu_runtime_seconds"]

# Bubble Plot (Side-by-Side CPU and GPU)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharex=True, sharey=True)

# Normalize sizes and colors across both plots
all_runtimes = pd.concat([CPU_Z, GPU_Z])
size_scale = 300 / all_runtimes.max()

sc1 = ax1.scatter(X, Y, s=CPU_Z * size_scale, c=CPU_Z, cmap="viridis", alpha=0.7, edgecolors="k")
sc2 = ax2.scatter(X, Y, s=GPU_Z * size_scale, c=GPU_Z, cmap="viridis", alpha=0.7, edgecolors="k")

# Set labels and titles
ax1.set_title("CPU Runtime")
ax2.set_title("GPU Runtime")
for ax in [ax1, ax2]:
    ax.set_xlabel("Number of Edges")
    ax.set_ylabel("Number of Parallel Simulations")
    ax.grid(True)

# Shared colorbar
cbar = fig.colorbar(sc2, ax=[ax1, ax2], orientation='vertical', fraction=0.03, pad=0.04)
cbar.set_label("Runtime (seconds)")

plt.suptitle("Bubble Plot: CPU vs GPU Runtime", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("bubble_plot_side_by_side.png")
plt.show()

# 3D Surface Plot for CPU and GPU
# Reshape data into grids
x_unique = sorted(df["num_edges"].unique())
y_unique = sorted(df["num_parallel_simulations"].unique())
X_grid, Y_grid = np.meshgrid(x_unique, y_unique)

CPU_Z_grid = np.full_like(X_grid, np.nan, dtype=float)
GPU_Z_grid = np.full_like(X_grid, np.nan, dtype=float)

for i, y_val in enumerate(y_unique):
    for j, x_val in enumerate(x_unique):
        match = df[(df["num_edges"] == x_val) & (df["num_parallel_simulations"] == y_val)]
        if not match.empty:
            CPU_Z_grid[i, j] = match.iloc[0]["cpu_runtime_seconds"]
            GPU_Z_grid[i, j] = match.iloc[0]["gpu_runtime_seconds"]

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

surf1 = ax.plot_surface(X_grid, Y_grid, CPU_Z_grid, cmap=cm.viridis, edgecolor='none', alpha=0.85)
surf2 = ax2.plot_surface(X_grid, Y_grid, GPU_Z_grid, cmap=cm.viridis, edgecolor='none', alpha=0.85)

for a, title in zip([ax, ax2], ["CPU Runtime Surface", "GPU Runtime Surface"]):
    a.set_xlabel("Number of Edges")
    a.set_ylabel("Number of Parallel Simulations")
    a.set_zlabel("Runtime (seconds)")
    a.set_title(title)

fig.colorbar(surf2, ax=[ax, ax2], orientation='vertical', fraction=0.02, pad=0.1, label='Runtime (seconds)')
plt.tight_layout()
plt.savefig("3d_surface_side_by_side.png")
plt.show()
