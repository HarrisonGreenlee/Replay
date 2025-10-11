import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.ticker import MaxNLocator

def plot_summary(summary_file):
    df = pd.read_csv(summary_file)
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df.dropna(subset=['time'], inplace=True)
    df = df.sort_values(['simulation_id', 'time'])
    min_time = df['time'].min()
    df['delta_time'] = df['time'] - min_time

    if df['delta_time'].nunique() <= 1:
        print("Not enough variation in time data to plot.")
        return

    # ---- Determine time scale ----
    max_time = df['delta_time'].max()

    units = [
        ('seconds', 1),
        ('minutes', 60),
        ('hours', 3600),
        ('days', 86400),
        ('months', 30 * 86400),
        ('years', 365 * 86400)
    ]

    # Choose the largest unit where scaled time span > 1
    chosen_unit, scale_factor = next(
        ((name, val) for name, val in reversed(units) if max_time / val >= 1),
        ('seconds', 1)
    )

    df['scaled_time'] = df['delta_time'] / scale_factor
    max_scaled_time = max_time / scale_factor

    # ---- Setup plot ----
    plt.figure(figsize=(10, 5))

    # ---- Colors, labels, and fixed legend order ----
    STATE_COLORS = {
        'susceptible': 'blue',
        'exposed': 'orange',
        'infectious': 'red',
        'resistant': 'green',
    }
    # Desired legend/display order (capitalized as requested)
    DISPLAY_ORDER = ['susceptible', 'exposed', 'infectious', 'resistant']
    LABELS = {
        'susceptible': 'Susceptible',
        'exposed': 'Exposed',
        'infectious': 'Infectious',
        'resistant': 'Resistant',
    }

    # Only include states present in the CSV, but keep the requested order
    available_states = [s for s in DISPLAY_ORDER if s in df.columns]
    if not available_states:
        print("No recognized state columns found in the CSV.")
        return

    sim_ids = df['simulation_id'].unique()
    for sim_id in sim_ids:
        sim_df = df[df['simulation_id'] == sim_id]
        for state in available_states:
            plt.plot(
                sim_df['scaled_time'],
                sim_df[state],
                label=LABELS[state] if sim_id == sim_ids[0] else None,  # capitalized legend labels
                color=STATE_COLORS[state],
                alpha=0.1
            )

    # ---- Auto ticks and labels ----
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', prune='both'))
    ax.set_xlim(0, max_scaled_time)

    plt.xlabel(f"Time ({chosen_unit})")
    plt.ylabel("Number of individuals")
    plt.title("Disease State Over Time Across Simulations")
    plt.ylim(0, df[available_states].values.max() * 1.05)
    
    leg = plt.legend(title="States", labelcolor='black',
                 facecolor='white', framealpha=1, edgecolor='0.2')

    # Make the little lines in the legend fully opaque & a bit thicker
    for h in leg.legend_handles:
        if hasattr(h, "set_alpha"):
            h.set_alpha(1)          # override your plot alpha (e.g., 0.1) just for legend
        if hasattr(h, "set_linewidth"):
            h.set_linewidth(2.2)    # thicker legend swatch

    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print(f"Usage: python {os.path.basename(__file__)} <summary_csv_file>")
        sys.exit(1)
    plot_summary(sys.argv[1])
