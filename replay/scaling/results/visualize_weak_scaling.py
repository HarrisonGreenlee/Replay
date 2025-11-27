import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os

OUTPUT_DIR = "plots"
CSV_FILE = "weak_scaling_summary.csv"

# Configure for PGF (LaTeX) output with serif fonts
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "pgf.rcfonts": False,
})

def thousands_formatter(x, pos):
    """
    Format large numbers in thousands using 'k'.
    For example: 40000 -> 40k, 8000 -> 8k.
    """
    if x >= 1000 or x <= -1000:
        return f"{int(x / 1000)}k"
    return f"{int(x)}"

def visualize_weak_scaling(csv_path: str = CSV_FILE):
    """
    Reads the weak_scaling_summary.csv file and plots
    Parallel Simulations (chosen_M) vs Percentage of GPU Cores.
    Saves PNG, PDF, and PGF versions of the figure.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Keep rows where percentage and chosen_M are present
    df = df.dropna(subset=["percentage", "chosen_M"]).copy()
    df["percentage"] = df["percentage"].astype(int)
    df = df.sort_values("percentage")

    # Create output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Plot
    plt.figure(figsize=(10, 6))

    LARGE_FONT = 29

    # Dashed line with circle markers, to match previous style
    plt.plot(
        df["percentage"].to_numpy(),
        df["chosen_M"].to_numpy(),
        "o--",
        label="Parallel Simulations",
    )

    # Apply thousands formatter on Y axis
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    # Labels and title
    plt.title("Weak Scaling: Simulation Parallelism", fontsize=LARGE_FONT)
    plt.xlabel("Percentage of GPU Cores", fontsize=LARGE_FONT)
    plt.ylabel("Parallel Simulations", fontsize=LARGE_FONT)

    plt.xticks(df["percentage"].unique(), fontsize=LARGE_FONT)
    plt.yticks(fontsize=LARGE_FONT)
    plt.tick_params(axis="y", pad=10)
    plt.legend(fontsize=16)
    plt.tight_layout()

    # Save in multiple formats
    base_path = os.path.join(OUTPUT_DIR, "weak_scaling_parallelism")

    png_file = f"{base_path}.png"
    plt.savefig(png_file, bbox_inches="tight")
    print(f"[INFO] Saved PNG to {png_file}")

    pdf_file = f"{base_path}.pdf"
    plt.savefig(pdf_file, bbox_inches="tight")
    print(f"[INFO] Saved PDF to {pdf_file}")

    pgf_file = f"{base_path}.pgf"
    plt.savefig(pgf_file, bbox_inches="tight")
    print(f"[INFO] Saved PGF to {pgf_file}")

    plt.close()

def main():
    visualize_weak_scaling()

if __name__ == "__main__":
    main()

