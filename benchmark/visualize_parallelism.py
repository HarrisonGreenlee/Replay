import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter

# Set PGF-compatible rendering options globally
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    # "text.usetex": True,
    "pgf.rcfonts": False,
})

CSV_FILE = "parallelism_benchmark.csv"
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualize_gpu_benchmark():
    if not os.path.exists(CSV_FILE):
        print(f"[WARNING] {CSV_FILE} not found. Exiting.")
        return

    df = pd.read_csv(CSV_FILE)
    df = df.dropna(subset=["SIM_PARALLEL_RUNS", "runtime_seconds"])
    df.sort_values(by="SIM_PARALLEL_RUNS", inplace=True)

    # Convert SIM_PARALLEL_RUNS to millions
    df["SIM_PARALLEL_RUNS_M"] = df["SIM_PARALLEL_RUNS"] / 1_000_000

    plt.figure(figsize=(10, 6))
    plt.plot(
        df["SIM_PARALLEL_RUNS_M"],
        df["runtime_seconds"],
        marker="o",
        linestyle="-",
        linewidth=1.5,
        label="GPU Runtime"
    )

    plt.title("GPU Runtime as a Function of Parallelism (Villages Scenario, Pop. 1000)")
    plt.xlabel("Number of Parallel Simulations")
    plt.ylabel("Runtime (seconds)")

    # Smart formatter: decimals only if needed
    def millions_formatter(x, pos):
        return f"{x:.0f} M" if x.is_integer() else f"{x:.2f} M"

    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    # Save plots
    png_file = os.path.join(OUTPUT_DIR, "parallelism_benchmark.png")
    pgf_file = os.path.join(OUTPUT_DIR, "parallelism_benchmark.pgf")

    plt.savefig(png_file)
    print(f"[INFO] Saved PNG to {png_file}")

    plt.savefig(pgf_file)
    print(f"[INFO] Saved PGF to {pgf_file}")

    plt.close()

if __name__ == "__main__":
    visualize_gpu_benchmark()
