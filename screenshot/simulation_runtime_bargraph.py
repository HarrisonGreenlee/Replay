import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Scale factor for text
SCALE = 2

# LaTeX-style look without external TeX install
rcParams.update({
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    # Double the default font size for all text that doesn't set fontsize explicitly
    "font.size": 10 * SCALE,          # default is ~10
    "axes.titlesize": 12 * SCALE,     # default ~12
    "axes.labelsize": 10 * SCALE,     # default ~10
    "xtick.labelsize": 10 * SCALE,
    "ytick.labelsize": 10 * SCALE,
    "legend.fontsize": 10 * SCALE,
})

# Data
tools = ["REPLAY", "NDlib", "Tacoma"]
seconds = [26, 2578, 26]

# Sort ascending
pairs = sorted(zip(seconds, tools), reverse=True)
sorted_seconds, sorted_tools = zip(*pairs)
ypos = np.arange(len(sorted_tools))

plt.figure(figsize=(6.75, 3.5), dpi=600)
ax = plt.gca()

# Bars (textured)
bars = ax.barh(
    ypos, sorted_seconds, height=0.78,
    facecolor="0.7", edgecolor="0.2", linewidth=1.8,
    hatch="////"
)

# Title and axis label (indicate log scale) — explicitly doubled
#ax.set_title(r"$\mathrm{Simulation\ Runtimes}$", pad=8, fontsize=26)
ax.set_xlabel(r"$\mathrm{Runtime\ in\ Seconds\ (log\ scale)}$", labelpad=6, fontsize=26)

# Log scale; hide ticks and bottom spine
ax.set_xscale("log")
ax.set_xlim(10, 3000)
ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
ax.spines["left"].set_visible(True)
ax.spines["left"].set_linewidth(1.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)

# Hide y-axis; annotate tool names on the left (make them bigger)
ax.set_yticks([])
ax.yaxis.set_visible(False)
for y, name in zip(ypos, sorted_tools):
    ax.annotate(rf"$\mathrm{{{name}}}$",
                xy=(10, y), xytext=(-10, 0), textcoords="offset points",
                ha="right", va="center",
                fontsize=20)  # doubled

# Center numerical labels without commas or "s" — doubled from 13 -> 26
for y, val in zip(ypos, sorted_seconds):
    x_center = (10 * val) ** 0.5
    ax.annotate(
        rf"${val}$",
        xy=(x_center, y),
        xytext=(0, 0),
        textcoords="offset points",
        ha="center", va="center",
        fontsize=26,  # doubled
        bbox=dict(facecolor="white", edgecolor="0.3", boxstyle="round,pad=0.2", alpha=0.95)
    )

plt.tight_layout()
png_path = "sim_runtimes.png"
plt.savefig(png_path, bbox_inches="tight", dpi=600)
plt.show()
