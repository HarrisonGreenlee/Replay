#ifndef GRAPH_GENERATOR_H
#define GRAPH_GENERATOR_H

#include <time.h>

void generate_graph(time_t base_time, int steps, int step_size,
                    int population_size, int num_groups,
                    double probability_inner_group_contact,
                    double probability_inter_group_contact,
                    const char *output_file, int seed);

#endif
