#!/usr/bin/env python3
from gooey import Gooey, GooeyParser
import subprocess
from datetime import datetime
import os
import sys

# Fix blurry text on high-DPI displays (Windows only)
if sys.platform == "win32":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def extract_population_size(path):
    try:
        with open(path, 'r') as f:
            first_line = f.readline()
        if not first_line.startswith("EDGE_LIST"):
            raise ValueError("Not a valid temporal contact file.")
        tokens = first_line.strip().split()
        return len(tokens) - 1
    except Exception as e:
        print(f"[ERROR] Failed to read population size: {e}")
        return None


@Gooey(
    program_name="Replay!",
    default_size=(1200, 1400),
    advanced=True,
    show_success_modal=False,
    show_stop_warning=True,
    tabbed_groups=True,
    terminal_font_color='white',
    terminal_background_color='black',
    requires_shell=False,
    clear_before_run=True
)
def main():
    parser = GooeyParser(description="Run a GPU/CPU simulation of temporal contact-based epidemics")

    files_group = parser.add_argument_group("Files", gooey_options={'columns': 1})

    files_group.add_argument("temporal_contact_file", widget="FileChooser",
                            help="Select temporal contact file",
                            gooey_options={'column_span': 2})  # Span full width (forces next to new row)

    files_group.add_argument("--summary-out", widget="FileSaver", help="CSV output file with SEIR counts for each simulation step")

    files_group.add_argument("--visualize", action="store_true", help="Visualize SEIR output")

    files_group.add_argument("--node-state-out", widget="FileSaver", help="CSV output file with full node-state tracking")

    sim_group = parser.add_argument_group("Simulation Configuration", gooey_options={'columns': 2})

    sim_group.add_argument("--use-cpu", action="store_true", help="Run simulation on CPU instead of GPU")

    sim_group.add_argument("--cpu-threads", type=int, default=1,
                           help="Number of CPU threads to use")

    sim_group.add_argument("--M", type=int, default=1000, help="Number of parallel simulations")
    sim_group.add_argument("--step-size", type=float, default=3600.0, help="Time step size (seconds)")
    sim_group.add_argument("--iterations", type=int, default=42, help="Number of simulation steps")

    sim_group.add_argument("--initial-infected", type=float, default=0.5,
                           help="Initial infection probability [0–1]")
    sim_group.add_argument("--infect-prob", type=float, default=0.99,
                           help="Infection probability when exposed for a full time step [0–1]")
    sim_group.add_argument("--upper-range", type=int, default=7200,
                           help="Incubation phase threshold (seconds)")
    sim_group.add_argument("--medium-range", type=int, default=3600,
                           help="Infectious phase threshold (seconds)")
    sim_group.add_argument("--lower-range", type=int, default=0,
                           help="Resistant phase threshold (seconds)")

    sim_group.add_argument("--start-date", widget="DateChooser",
                           default="2000-01-01", help="Simulation start date (UTC)")
    sim_group.add_argument("--start-time", default="00:00:00", help="Simulation start time (HH:MM:SS, UTC)")
    sim_group.add_argument("--time-step", type=int, default=3600, help="Iteration timestep (seconds)")

    args = parser.parse_args()

    start_str = f"{args.start_date}T{args.start_time}"
    try:
        datetime.strptime(args.start_time, "%H:%M:%S")
        start_dt = datetime.fromisoformat(start_str)
        start_ts = int(start_dt.timestamp())
    except ValueError:
        print(f"[ERROR] Invalid datetime format: {start_str}. Use YYYY-MM-DD and HH:MM:SS.")
        return

    population_size = extract_population_size(args.temporal_contact_file)
    if not population_size:
        print("[ERROR] Could not determine population size from temporal contact file.")
        return

    sim_binary = "gpu_cpu_temporal_sim.exe" if os.name == 'nt' else "./gpu_cpu_temporal_sim"
    cmd = [sim_binary, args.temporal_contact_file]

    if args.use_cpu:
        cmd.append("--cpu-only")
        cmd += ["--cpu-threads", str(args.cpu_threads)]

    cmd += [
        "--N", str(population_size),
        "--M", str(args.M),
        "--step-size", str(args.step_size),
        "--iterations", str(args.iterations),
        "--initial-infected", str(args.initial_infected),
        "--infect-prob", str(args.infect_prob),
        "--upper-range", str(args.upper_range),
        "--medium-range", str(args.medium_range),
        "--lower-range", str(args.lower_range),
        "--start-time", str(start_ts),
        "--time-step", str(args.time_step),
    ]

    if args.summary_out:
        cmd += ["--summary-out", args.summary_out]
    if args.node_state_out:
        cmd += ["--node-state-out", args.node_state_out]

    print("Executing simulation with command:")
    print(" ".join(cmd))
    print("-" * 60)

    subprocess.run(cmd)

    if args.visualize and args.summary_out:
        try:
            from visualize_summary import plot_summary
            print("[INFO] Launching visualization...")
            plot_summary(args.summary_out)
        except Exception as e:
            print(f"[WARNING] Could not visualize results: {e}")


if __name__ == "__main__":
    main()
