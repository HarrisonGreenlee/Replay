import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def plot_summary(summary_file):
    df = pd.read_csv(summary_file)
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df.dropna(subset=['time'], inplace=True)
    df = df.sort_values(['simulation_id', 'time'])
    min_time = df['time'].min()
    df['delta_time'] = df['time'] - min_time

    time_diffs = np.diff(sorted(df['delta_time'].unique()))
    step_size = np.min(time_diffs[time_diffs > 0])

    max_time = df['delta_time'].max()
    max_y = df[['exposed', 'infectious', 'resistant', 'susceptible']].values.max()

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
                sim_df['delta_time'],
                sim_df[state],
                label=state if sim_id == sim_ids[0] else None,
                color=state_colors[state],
                alpha=0.1
            )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Number of individuals")
    plt.title("Disease State Over Time Across Simulations")
    plt.xlim(0, max_time)
    plt.ylim(0, max_y * 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print(f"Usage: python {os.path.basename(__file__)} <summary_csv_file>")
        sys.exit(1)
    plot_summary(sys.argv[1])
