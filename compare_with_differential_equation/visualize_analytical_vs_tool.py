def plot_summary(summary_file, seir_csv=None, output_dir="plots", save_pgf=True):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.lines import Line2D

    if save_pgf:
        plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": [],
            "axes.labelsize": 10,
            "font.size": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pgf.rcfonts": False,
        })

    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess data
    df = pd.read_csv(summary_file)
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df.dropna(subset=['time'], inplace=True)
    df = df.sort_values(['simulation_id', 'time'])
    min_time = df['time'].min()
    df['delta_days'] = (df['time'] - min_time) / 86400
    max_time = df['delta_days'].max()
    max_y = df[['exposed', 'infectious', 'resistant', 'susceptible']].values.max()

    plt.figure(figsize=(7, 4))  # clearer resolution

    # Colors
    state_colors = {
        'exposed': 'orange',
        'infectious': 'red',
        'resistant': 'green',
        'susceptible': 'blue'
    }

    # Plot faint simulation runs
    for sim_id in df['simulation_id'].unique():
        sim_df = df[df['simulation_id'] == sim_id]
        for state in state_colors:
            plt.plot(
                sim_df['delta_days'],
                sim_df[state],
                color=state_colors[state],
                alpha=0.05,
                linewidth=0.25
            )

    # Plot analytical SEIR if available
    if seir_csv:
        seir_df = pd.read_csv(seir_csv)
        label_map = {
            'Susceptible': 'Analytical S',
            'Exposed': 'Analytical E',
            'Infectious': 'Analytical I',
            'Recovered': 'Analytical R'
        }
        color_map = {
            'Susceptible': 'darkblue',
            'Exposed': 'darkorange',
            'Infectious': 'darkred',
            'Recovered': 'darkgreen'
        }

        for col in label_map:
            if col in seir_df.columns:
                plt.plot(
                    seir_df['Day'],
                    seir_df[col],
                    color=color_map[col],
                    linestyle='--',
                    linewidth=2.5,
                    label=label_map[col]
                )

    # Labels and layout
    plt.xlabel("Time (days)")
    plt.ylabel("Number of individuals")
    plt.title("Disease State Over Time")
    # plt.xlim(0, max_time)
    plt.xlim(0, 2)
    plt.ylim(0, max_y * 1.05)
    plt.grid(True)
    plt.tight_layout()

    # Create clear, thick legend handles
    legend_elements = [
        Line2D([0], [0], color=state_colors['susceptible'], lw=2, label='Susceptible'),
        Line2D([0], [0], color=state_colors['exposed'], lw=2, label='Exposed'),
        Line2D([0], [0], color=state_colors['infectious'], lw=2, label='Infectious'),
        Line2D([0], [0], color=state_colors['resistant'], lw=2, label='Recovered'),
        Line2D([0], [0], color='darkblue', linestyle='--', lw=2.5, label='Analytical S'),
        Line2D([0], [0], color='darkorange', linestyle='--', lw=2.5, label='Analytical E'),
        Line2D([0], [0], color='darkred', linestyle='--', lw=2.5, label='Analytical I'),
        Line2D([0], [0], color='darkgreen', linestyle='--', lw=2.5, label='Analytical R'),
    ]
    plt.legend(handles=legend_elements, loc="upper right")

    # Save files
    base = "compare_with_compartmental"
    for ext in ["png", "pdf"] + (["pgf"] if save_pgf else []):
        path = os.path.join(output_dir, f"{base}.{ext}")
        if ext == "png":
            plt.savefig(path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(path, bbox_inches='tight')
        print(f"[INFO] Saved {ext.upper()} to {path}")

    plt.close()


plot_summary("out.csv", "seir_output.csv")