import pandas as pd
import matplotlib.pyplot as plt
import os

# Define scenarios and output directory
SCENARIOS = ["villages", "cities", "airport", "dense"]
OUTPUT_DIR = "plots"

# Configure for PGF (LaTeX) output
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "pgf.rcfonts": False,
})

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _compute_stats(df: pd.DataFrame, runtime_col: str) -> pd.DataFrame:
    """
    For a given runtime column (e.g., 'cpu_runtime' or 'gpu_runtime'),
    compute per-population mean/min/max across trials and the asymmetric
    errors needed for error bars.
    """
    # Keep only rows that have a value for the requested runtime
    d = df.dropna(subset=[runtime_col])

    if d.empty:
        return pd.DataFrame(columns=["population", "mean", "min", "max", "lower_err", "upper_err"])

    stats = (
        d.groupby("population")[runtime_col]
        .agg(mean="mean", min="min", max="max")
        .reset_index()
        .sort_values("population")
    )
    stats["lower_err"] = stats["mean"] - stats["min"]
    stats["upper_err"] = stats["max"] - stats["mean"]
    return stats

def visualize_scenario(scenario):
    """
    Reads the CSV for a given scenario, plots mean CPU vs. mean GPU runtimes
    with asymmetric error bars (min/max across trials) on one graph, and saves
    PNG, PDF, and PGF figures into the output directory.
    """
    csv_file = f"{scenario}.csv"

    if not os.path.exists(csv_file):
        print(f"[WARNING] {csv_file} not found. Skipping.")
        return

    # Read CSV into a DataFrame
    df = pd.read_csv(csv_file)

    # Keep rows where population is present; ensure integer type for sorting/labels
    df = df.dropna(subset=["population"]).copy()
    df["population"] = df["population"].astype(int)

    # --- Aggregate over trials -------------------------------------------------
    cpu_stats = _compute_stats(df, "cpu_runtime")
    gpu_stats = _compute_stats(df, "gpu_runtime")

    # Plot
    plt.figure(figsize=(10, 6))

    # CPU: mean with asymmetric min/max error bars
    if not cpu_stats.empty:
        plt.errorbar(
            cpu_stats["population"].to_numpy(),
            cpu_stats["mean"].to_numpy(),
            yerr=[cpu_stats["lower_err"].to_numpy(), cpu_stats["upper_err"].to_numpy()],
            fmt="o--",            # marker + dashed line for the mean
            capsize=5,            # error bar caps
            label="CPU (mean ± min/max)",
        )

    # GPU: mean with asymmetric min/max error bars
    if not gpu_stats.empty:
        plt.errorbar(
            gpu_stats["population"].to_numpy(),
            gpu_stats["mean"].to_numpy(),
            yerr=[gpu_stats["lower_err"].to_numpy(), gpu_stats["upper_err"].to_numpy()],
            fmt="s--",            # different marker for GPU
            capsize=5,
            label="GPU (mean ± min/max)",
        )

    # Increase font sizes
    LARGE_FONT = 29
    plt.title(f"Runtime Comparison for {scenario.capitalize()} Scenario", fontsize=LARGE_FONT)
    plt.xlabel("Population Size", fontsize=LARGE_FONT)
    plt.ylabel("Runtime (seconds)", fontsize=LARGE_FONT)
    plt.xticks(fontsize=LARGE_FONT)
    plt.yticks(fontsize=LARGE_FONT)
    plt.tick_params(axis='y', pad=10)
    plt.legend(fontsize=16)
    plt.tight_layout()

    # Save PNG
    png_file = os.path.join(OUTPUT_DIR, f"{scenario}_runtime_comparison.png")
    plt.savefig(png_file, bbox_inches='tight')
    print(f"[INFO] Saved PNG to {png_file}")

    # Save PDF
    pdf_file = os.path.join(OUTPUT_DIR, f"{scenario}_runtime_comparison.pdf")
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"[INFO] Saved PDF to {pdf_file}")

    # Save PGF
    pgf_file = os.path.join(OUTPUT_DIR, f"{scenario}_runtime_comparison.pgf")
    plt.savefig(pgf_file, bbox_inches='tight')
    print(f"[INFO] Saved PGF to {pgf_file}")

    plt.close()

def main():
    for scenario in SCENARIOS:
        visualize_scenario(scenario)

if __name__ == "__main__":
    main()
