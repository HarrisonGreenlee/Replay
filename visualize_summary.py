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
    plt.figure(figsize=(12, 6))
    state_colors = {
        'exposed': 'orange',
        'infectious': 'red',
        'resistant': 'green',
        'susceptible': 'blue'
    }

    sim_ids = df['simulation_id'].unique()
    for sim_id in sim_ids:
        sim_df = df[df['simulation_id'] == sim_id]
        for state in state_colors:
            plt.plot(
                sim_df['scaled_time'],
                sim_df[state],
                label=state if sim_id == sim_ids[0] else None,
                color=state_colors[state],
                alpha=0.1
            )

    # ---- Auto ticks and labels ----
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', prune='both'))
    ax.set_xlim(0, max_scaled_time)

    plt.xlabel(f"Time ({chosen_unit})")
    plt.ylabel("Number of individuals")
    plt.title("Disease State Over Time Across Simulations")
    plt.ylim(0, df[['exposed', 'infectious', 'resistant', 'susceptible']].values.max() * 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print(f"Usage: python {os.path.basename(__file__)} <summary_csv_file>")
        sys.exit(1)
    plot_summary(sys.argv[1])
