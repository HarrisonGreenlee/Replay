#include "graph_generator.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/**
 * Helper function to generate a timestamp string in ISO 8601 format (UTC).
 *  - base_time: the starting time (from command line)
 *  - step: current step index
 *  - step_size: number of seconds per step
 *  - buffer: where to store the resulting string in ISO 8601 format
 */
void generate_timestamp(time_t base_time, int step, int step_size,
                        char *buffer) {
  // Offset by (step * step_size) plus a random offset within [0, step_size).
  time_t timestamp = base_time + (step * step_size) + (rand() % step_size);
  struct tm *tm_info = gmtime(&timestamp);
  if (strftime(buffer, 21, "%Y-%m-%dT%H:%M:%SZ", tm_info) == 0) {
    fprintf(stderr, "Warning: Failed to generate timestamp.\n");
    strcpy(buffer, "1970-01-01T00:00:00Z"); // fallback
  }
}

/**
 * Generates the temporal graph in a simplified text format:
 *
 *     EDGE_LIST 1 2 3 ... N
 *     <source>, <target>, <start_ts>, <end_ts>;
 *     ...
 *
 * - No large arrays are stored in memory.
 * - We write edges directly to a temporary file line by line.
 * - We keep a running count of edges in an integer counter.
 * - At the end, we create the final file, prepend the header, and then copy
 *   the edges from the temporary file.
 */
void generate_graph(time_t base_time, int steps, int step_size,
                    int population_size, int num_groups,
                    double probability_inner_group_contact,
                    double probability_inter_group_contact,
                    const char *output_file, int seed) {
  // 1) Seed the random generator
  if (seed < 0) {
    fprintf(stderr, "Seed must be a non-negative integer.\n");
    return;
  }
  srand((unsigned int)seed);

  // 2) Prepare group assignments (round-robin)
  int *group_assignment = (int *)malloc(population_size * sizeof(int));
  for (int i = 0; i < population_size; i++) {
    group_assignment[i] = i % num_groups;
  }

  // 3) Create a temporary file to store edges as we generate them
  char tmp_filename[256];
  snprintf(tmp_filename, sizeof(tmp_filename), "%s.tmp", output_file);

  FILE *tmp_fp = fopen(tmp_filename, "w");
  if (!tmp_fp) {
    fprintf(stderr,
            "Error: Could not open temporary file '%s' for writing: %s\n",
            tmp_filename, strerror(errno));
    free(group_assignment);
    return;
  }

  // 4) Generate edges, write each immediately to the temporary file
  //    Keep track of the total edge count in `long long` for large graphs
  long long edge_count = 0;
  char buffer_start[21], buffer_end[21];

  for (int step = 0; step < steps; step++) {
    // Optional progress indicator
    printf("Generating step %d/%d...\n", step + 1, steps);

    for (int i = 0; i < population_size; i++) {
      for (int j = i + 1; j < population_size; j++) {
        int same_group = (group_assignment[i] == group_assignment[j]);
        double probability = same_group ? probability_inner_group_contact
                                        : probability_inter_group_contact;

        if (((double)rand() / RAND_MAX) < probability) {
          // Generate start/end timestamps
          generate_timestamp(base_time, step, step_size, buffer_start);
          generate_timestamp(base_time, step, step_size, buffer_end);

          // Ensure buffer_start <= buffer_end
          if (strcmp(buffer_start, buffer_end) > 0) {
            char temp[21];
            strcpy(temp, buffer_start);
            strcpy(buffer_start, buffer_end);
            strcpy(buffer_end, temp);
          }

          // Write the edge line to the temporary file
          // Use 1-based node IDs in the final output
          fprintf(tmp_fp, "%d, %d, %s, %s;\n", i + 1, j + 1, buffer_start,
                  buffer_end);

          edge_count++;
        }
      }
    }
  }

  fclose(tmp_fp);
  free(group_assignment);

  // 5) Now we create the final file, write the headers, then append the edges
  FILE *final_fp = fopen(output_file, "w");
  if (!final_fp) {
    fprintf(stderr, "Error: Could not open output file '%s' for writing: %s\n",
            output_file, strerror(errno));
    // We won't remove the temp file so data isn't lost
    return;
  }

  // Write: "NODE_LIST 1 2 3 ... N"
  fprintf(final_fp, "NODE_LIST");
  for (int i = 1; i <= population_size; i++) {
    fprintf(final_fp, " %d", i);
  }
  fprintf(final_fp, "\n");

  // Write: "NUM_EDGES <count>"
  // REMOVED FROM FILE FORMAT, SO SKIP THIS
  // fprintf(final_fp, "NUM_EDGES %lld\n", edge_count);

  // Copy all lines from the temporary file into final file
  FILE *read_tmp = fopen(tmp_filename, "r");
  if (!read_tmp) {
    fclose(final_fp);
    fprintf(stderr, "Error: Failed to re-open temp file '%s': %s\n",
            tmp_filename, strerror(errno));
    return;
  }

  // Append lines from the temp file
  char line_buf[1024];
  while (fgets(line_buf, sizeof(line_buf), read_tmp)) {
    fputs(line_buf, final_fp);
  }

  fclose(read_tmp);
  fclose(final_fp);

  // 6) Clean up: remove the temp file
  if (remove(tmp_filename) != 0) {
    fprintf(stderr, "Warning: Could not remove temp file '%s': %s\n",
            tmp_filename, strerror(errno));
  }

  printf("Temporal graph file generated: %s\n", output_file);
}
