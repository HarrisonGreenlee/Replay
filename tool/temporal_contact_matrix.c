#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "intervaldb.h"
#include "temporal_contact_matrix.h"
#include "uthash.h"

typedef struct {
  int row;
  int col;
  float weight;
  UT_hash_handle hh;
} EdgeAggEntry;

#define MAX_LINE_LEN 100000 // Max size for non-header lines
#define DEBUG_PRINT_COUNT 5 // Number of edges to print for verification

#ifdef _WIN32
typedef long ssize_t;
#endif

static int gTotalNodes = -1; // <-- parsed from NODE_LIST= line

// -----------------------------------------------------------------------------
// Utility: portable getline() replacement
// -----------------------------------------------------------------------------

ssize_t portable_getline(char **lineptr, size_t *n, FILE *stream) {
    if (!lineptr || !n || !stream) {
        errno = EINVAL;
        return -1;
    }

    if (*lineptr == NULL || *n == 0) {
        *n = 128;
        *lineptr = (char *)malloc(*n);
        if (*lineptr == NULL) {
            return -1;
        }
    }

    size_t i = 0;
    int c = 0;

    while ((c = fgetc(stream)) != EOF) {
        // grow buffer if needed
        if (i + 1 >= *n) {
            size_t new_size = *n * 2;
            char *new_ptr = realloc(*lineptr, new_size);
            if (!new_ptr) {
                return -1;
            }
            *lineptr = new_ptr;
            *n = new_size;
        }

        (*lineptr)[i++] = (char)c;
        if (c == '\n') {
            break;
        }
    }

    if (i == 0 && c == EOF) {
        return -1; // EOF with no data read
    }

    (*lineptr)[i] = '\0';
    return (ssize_t)i;
}

// -----------------------------------------------------------------------------
// Utility: pack/unpack node pairs into a 64-bit value
// -----------------------------------------------------------------------------

int64_t pack_node_pair(int a, int b) {
  // Enforce consistent ordering to avoid (i,j) ≠ (j,i) mismatch
  uint32_t min_id = a < b ? a : b;
  uint32_t max_id = a > b ? a : b;
  return ((int64_t)min_id << 32) | max_id;
}

void unpack_node_pair(int64_t packed, int *src, int *tgt) {
  *tgt = (int)(packed & 0xFFFFFFFF);
  *src = (int)((packed >> 32) & 0xFFFFFFFF);
}

// -----------------------------------------------------------------------------
// Timestamp parser (ISO8601 => Unix time)
// -----------------------------------------------------------------------------
int64_t parse_iso8601_to_unix(const char *time_str) {
  if (!time_str)
    return -1;

  struct tm t = {0};
  int year, month, day, hour, min, sec;
  char extra[10];

  if (sscanf(time_str, "%d-%d-%dT%d:%d:%d%9s", &year, &month, &day, &hour, &min,
             &sec, extra) < 6) {
    return -1;
  }

  t.tm_year = year - 1900;
  t.tm_mon = month - 1;
  t.tm_mday = day;
  t.tm_hour = hour;
  t.tm_min = min;
  t.tm_sec = sec;

#ifdef _WIN32
  return (int64_t)_mkgmtime(&t);
#else
  return (int64_t)timegm(&t);
#endif
}

int count_valid_edges(const char *filename) {
  FILE *fp = fopen(filename, "r");
  if (!fp)
    return -1;

  char line[MAX_LINE_LEN];
  int count = 0;

  while (fgets(line, sizeof(line), fp)) {
    if (line[0] == '#')
      continue; // skip comment lines
    int dummy1, dummy2;
    char s[64], e[64];
    if (sscanf(line, " %d, %d, %63[^,], %63[^;];", &dummy1, &dummy2, s, e) ==
        4) {
      count++;
    }
  }

  fclose(fp);
  return count;
}

// -----------------------------------------------------------------------------
// Build IntervalMap from file
// -----------------------------------------------------------------------------
IntervalMap *parse_edgelist_build_intervals(const char *filename,
                                            int *p_count) {
  FILE *fp_check = fopen(filename, "r");
  if (!fp_check) {
    fprintf(stderr, "[FATAL] Could not open file '%s'\n", filename);
    return NULL;
  }

  char *header = NULL;
  size_t header_bufsize = 0;
  ssize_t header_len = portable_getline(&header, &header_bufsize, fp_check);
  if (header_len <= 0) {
      fprintf(stderr, "[FATAL] Unable to read first line from file\n");
      free(header);
      fclose(fp_check);
      return NULL;
  }

  // printf("[DEBUG] Full header line:\n'%s'\n", header);
  printf("[DEBUG] Header string length: %zu\n", strlen(header));

  if (strncmp(header, "NODE_LIST", 9) != 0 ||
      (header[9] != ' ' && header[9] != '\t')) {
    fprintf(stderr,
            "[FATAL] First line must begin with 'NODE_LIST', got: '%s'\n",
            header);
    free(header);
    fclose(fp_check);
    return NULL;
  }

  // Tokenize after "NODE_LIST"
  char *token = strtok(header + 9, " \t\r\n");
  printf("[TRACE] Parsed token: '%s'\n", token);
  int node_id_count = 0;
  size_t node_capacity = 1024;
  int *node_ids = (int *)malloc(node_capacity * sizeof(int));
  if (!node_ids) {
    fprintf(stderr, "[FATAL] Memory allocation failed for node ID array\n");
    free(header);
    fclose(fp_check);
    return NULL;
  }

  while (token) {
    errno = 0;
    char *endptr = NULL;
    long id = strtol(token, &endptr, 10);

    if (errno != 0 || *endptr != '\0' || id < 0) {
      fprintf(stderr, "[FATAL] Invalid node ID in NODE_LIST: '%s'\n", token);
      free(node_ids);
      free(header);
      fclose(fp_check);
      return NULL;
    }

    // Duplicate check: simple linear scan for now
    for (int i = 0; i < node_id_count; i++) {
      if (node_ids[i] == id) {
        fprintf(stderr,
                "[FATAL] Duplicate node ID detected in NODE_LIST: %ld\n", id);
        free(node_ids);
        free(header);
        fclose(fp_check);
        return NULL;
      }
    }

    if (node_id_count >= node_capacity) {
        node_capacity *= 2;
        int *new_ids = realloc(node_ids, node_capacity * sizeof(int));
        if (!new_ids) {
            fprintf(stderr, "[FATAL] realloc failed for node ID array\n");
            free(node_ids);
            free(header);
            fclose(fp_check);
            return NULL;
        }
        node_ids = new_ids;
    }

    node_ids[node_id_count++] = (int)id;
    token = strtok(NULL, " \t\r\n");
  }

  if (node_id_count <= 0) {
    fprintf(stderr, "[FATAL] No valid node IDs found in NODE_LIST.\n");
    free(node_ids);
    free(header);
    fclose(fp_check);
    return NULL;
  }

  gTotalNodes = node_id_count;
  free(header);
  printf("[DEBUG] NODE_LIST parsed: gTotalNodes = %d\n", gTotalNodes);
  free(node_ids);

  if (!filename || !p_count)
    return NULL;

  int edge_count = count_valid_edges(filename);
  if (edge_count <= 0) {
    fprintf(stderr, "No valid edge lines found in %s\n", filename);
    return NULL;
  }

  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Error: could not open file '%s'\n", filename);
    return NULL;
  }

  IntervalMap *intervals = interval_map_alloc(edge_count);
  if (!intervals) {
    fclose(fp);
    return NULL;
  }

  char line[MAX_LINE_LEN];
  int interval_count = 0;

  while (fgets(line, sizeof(line), fp)) {
    if (line[0] == '#')
      continue; // skip comment lines

    int src_id, tgt_id;
    char start_str[64], end_str[64];

    int parsed = sscanf(line, " %d , %d , %63[^,] , %63[^;\r\n] ;", &src_id,
                        &tgt_id, start_str, end_str);
    if (parsed != 4) {
      fprintf(stderr, "Warning: malformed line skipped '%s'\n", line);
      continue;
    }

    int64_t start = parse_iso8601_to_unix(start_str);
    int64_t end = parse_iso8601_to_unix(end_str);
    if (start == -1 || end == -1)
      continue;

    char src_buf[12], tgt_buf[12];
    snprintf(src_buf, sizeof(src_buf), "%d", src_id);
    snprintf(tgt_buf, sizeof(tgt_buf), "%d", tgt_id);

    int src_index = get_or_create_node_index(src_buf);
    int tgt_index = get_or_create_node_index(tgt_buf);

    intervals[interval_count].start = start;
    intervals[interval_count].end = end;
    intervals[interval_count].target_id = pack_node_pair(src_index, tgt_index);
    intervals[interval_count].sublist = -1;

    if (interval_count < DEBUG_PRINT_COUNT) {
      printf("[DEBUG] Edge %d: %d -> %d | Start: %lld | End: %lld\n",
             interval_count, src_id, tgt_id, (long long)start, (long long)end);
    }

    interval_count++;
    if (interval_count > edge_count) {
      fprintf(stderr, "[FATAL] More edges parsed than expected (%d > %d)\n",
              interval_count, edge_count);
      exit(EXIT_FAILURE);
    }
  }

  fclose(fp);
  *p_count = interval_count;

  // printf("[INFO] Parsed %d temporal edges from '%s'\n", interval_count,
  // filename);
  return intervals;
}

// Node map (hash-based, using uthash)
static NodeItem *node_map = NULL;

// get_or_create_node_index:
//   Returns the compressed index of a node_id_str, creating one if needed.
int get_or_create_node_index(const char *node_id_str) {
  NodeItem *item = NULL;
  HASH_FIND_STR(node_map, node_id_str, item);
  if (!item) {
    item = (NodeItem *)malloc(sizeof(NodeItem));
    if (!item) {
      fprintf(stderr, "[FATAL] Out of memory in get_or_create_node_index\n");
      exit(EXIT_FAILURE);
    }

#ifdef _WIN32
    item->node_id = _strdup(node_id_str);
#else
    item->node_id = strdup(node_id_str);
#endif

    if (!item->node_id) {
      fprintf(stderr, "[FATAL] strdup failed for node ID: %s\n", node_id_str);
      exit(EXIT_FAILURE);
    }

    item->index = HASH_COUNT(node_map);
    HASH_ADD_KEYPTR(hh, node_map, item->node_id, strlen(item->node_id), item);

    printf("[DEBUG] New node added: %s -> index %d\n", item->node_id,
           item->index);
  }
  return item->index;
}

// get_total_node_count:
//   Gets the total number of nodes in the simulation.
int get_total_node_count(void) { return gTotalNodes; }

// free_node_map:
//   Frees the node_map and its resources.
void free_node_map(void) {
  NodeItem *current, *tmp;
  HASH_ITER(hh, node_map, current, tmp) {
    HASH_DEL(node_map, current);
    free(current->node_id);
    free(current);
  }
  node_map = NULL;
}

// -----------------------------------------------------------------------------
// NCLS Build/Cleanup
// -----------------------------------------------------------------------------
int build_interval_db_wrapper(IntervalMap *im, int n, IntervalDBWrapper *dbw) {
  if (!im || n <= 0)
    return 0;

  int n_compact = 0;
  int n_sublists = 0;
  SublistHeader *subh = build_nested_list(im, n, &n_compact, &n_sublists);
  if (!subh)
    return 0;

  dbw->im = im;
  dbw->subheader = subh;
  dbw->n = n_compact;
  dbw->nlists = n_sublists;
  return 1;
}

void free_interval_db_wrapper(IntervalDBWrapper *dbw) {
  if (dbw->subheader)
    free(dbw->subheader);
  if (dbw->im)
    free(dbw->im);
  dbw->subheader = NULL;
  dbw->im = NULL;
  dbw->n = 0;
  dbw->nlists = 0;
}
