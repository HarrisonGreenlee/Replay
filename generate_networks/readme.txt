Contact Network Simulation Tool - Command Line Usage

Invocation:

contact_network.exe <start_datetime> <steps> <step_size> <population_size> <num_groups> <prob_inner> <prob_inter> <output_file>

Parameters:

<start_datetime> (string) - Start time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS).

<steps> (int) - Number of simulation steps.

<step_size> (int) - Duration of each step in seconds.

<population_size> (int) - Total number of individuals in the simulation.

<num_groups> (int) - Number of distinct groups within the population.

<prob_inner> (double) - Probability (0.0 - 1.0) of interaction within the same group.

<prob_inter> (double) - Probability (0.0 - 1.0) of interaction between different groups.

<output_file> (string) - Path to the output graph file.

Example:

contact_network.exe "2000-01-01T00:00:00" 42 60 100 5 0.7 0.3 output.txt

Explanation of Example:

Simulation starts at Jan 1, 2000, at 00:00.

Runs for 42 steps, each step lasting 60 seconds.

Simulates 100 individuals, divided into 5 groups.

70% probability of contact within a group.

30% probability of contact between groups.

Outputs the result to output.txt

Output:

The tool generates a file containing nodes (individuals) and edges (interactions over time).

Each edge includes start and end timestamps representing when the contact occurred.

Notes:

Ensure that libxml2 is installed and properly linked for graph output.

Use double quotes around <start_datetime> to prevent shell interpretation issues.

Probabilities must be within the range 0.0 to 1.0.