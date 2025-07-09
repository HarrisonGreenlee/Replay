#include "graph_generator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Parses a datetime string in the format: "YYYY-MM-DDTHH:MM:SS"
time_t parse_datetime(const char *datetime_str) {
  int year, month, day, hour, minute, second;
  // Scan the string. Make sure the format matches exactly.
  // Example input: "2025-03-14T12:00:00"
  if (sscanf(datetime_str, "%d-%d-%dT%d:%d:%d", &year, &month, &day, &hour,
             &minute, &second) != 6) {
    fprintf(stderr, "Error parsing datetime string: %s\n", datetime_str);
    exit(EXIT_FAILURE);
  }

  struct tm tm_info;
  memset(&tm_info, 0, sizeof(tm_info));

  tm_info.tm_year = year - 1900; // struct tm year is "years since 1900"
  tm_info.tm_mon = month - 1;    // struct tm months are 0-based
  tm_info.tm_mday = day;
  tm_info.tm_hour = hour;
  tm_info.tm_min = minute;
  tm_info.tm_sec = second;

  // If you want local time interpretation:
  //   time_t t = mktime(&tm_info);
  // If you want UTC (i.e., ignoring local time zone):
  //   time_t t = _mkgmtime(&tm_info);
  // For this example, let's use local time:
  time_t t = mktime(&tm_info);

  return t;
}

int main(int argc, char *argv[]) {
  if (argc != 10) {
    printf("Usage: %s <start_datetime> <steps> <step_size> <population_size> "
           "<num_groups> "
           "<prob_inner> <prob_inter> <output_file> <seed>\n",
           argv[0]);
    return 1;
  }

  // 1) Parse the start datetime
  time_t base_time = parse_datetime(argv[1]);

  // 2) Parse numeric arguments
  int steps = atoi(argv[2]);
  int step_size = atoi(argv[3]);
  int population_size = atoi(argv[4]);
  int num_groups = atoi(argv[5]);
  double probability_inner_group_contact = atof(argv[6]);
  double probability_inter_group_contact = atof(argv[7]);
  char *output_file = argv[8];
  int seed = atoi(argv[9]);

  // 4) Call the generation function
  generate_graph(base_time, steps, step_size, population_size, num_groups,
                 probability_inner_group_contact,
                 probability_inter_group_contact, output_file, seed);

  return 0;
}
