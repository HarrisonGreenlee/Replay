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

def visualize_scenario(scenario):
    """
    Reads the CSV for a given scenario, plots CPU vs. GPU runtimes on one graph,
    and saves PNG, PDF, and PGF (LaTeX) figures into the output directory.
    """
    csv_file = f"{scenario}.csv"

    if not os.path.exists(csv_file):
        print(f"[WARNING] {csv_file} not found. Skipping.")
        return

    # Read CSV into a DataFrame
    df = pd.read_csv(csv_file)

    # Drop rows only if population is NaN
    df = df.dropna(subset=["population"])
    df["population"] = df["population"].astype(int)

    # Sort by population
    df.sort_values(by="population", inplace=True)

    # Split CPU and GPU data
    df_cpu = df.dropna(subset=["cpu_runtime"])
    df_gpu = df.dropna(subset=["gpu_runtime"])

    populations_cpu = df_cpu["population"].values
    cpu_times = df_cpu["cpu_runtime"].values

    populations_gpu = df_gpu["population"].values
    gpu_times = df_gpu["gpu_runtime"].values

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(populations_cpu, cpu_times, marker="o", linestyle="--", label="CPU")
    plt.plot(populations_gpu, gpu_times, marker="s", linestyle="--", label="GPU")

    # Increase font sizes
    LARGE_FONT = 29
    plt.title(f"Runtime Comparison for {scenario.capitalize()} Scenario", fontsize=LARGE_FONT)
    plt.xlabel("Population Size", fontsize=LARGE_FONT)
    plt.ylabel("Runtime (seconds)", fontsize=LARGE_FONT)
    plt.xticks(fontsize=LARGE_FONT)
    plt.yticks(fontsize=LARGE_FONT)
    plt.tick_params(axis='y', pad=10)
    plt.legend(fontsize=LARGE_FONT)

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
