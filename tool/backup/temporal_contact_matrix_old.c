#include "intervaldb.h"
#include "temporal_contact_matrix.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_LINE_LEN 512
#define DEBUG_PRINT_COUNT 5 // Number of edges to print for verification

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

// -----------------------------------------------------------------------------
// Build IntervalMap from new EDGE_LIST format
// -----------------------------------------------------------------------------
IntervalMap *parse_edgelist_build_intervals(const char *filename,
                                            int *p_count) {
  if (!filename || !p_count)
    return NULL;

  FILE *fp = fopen(filename, "r");
  if (!fp)
    return NULL;

  char line[MAX_LINE_LEN];

  // Skip EDGE_LIST line
  do {
    if (!fgets(line, sizeof(line), fp)) {
      fclose(fp);
      return NULL;
    }
  } while (strncmp(line, "EDGE_LIST", 9) != 0);

  // Read NUM_EDGES line
  do {
    if (!fgets(line, sizeof(line), fp)) {
      fclose(fp);
      return NULL;
    }
  } while (strncmp(line, "NUM_EDGES", 9) != 0);

  int expected_edges = 0;
  if (sscanf(line, "NUM_EDGES %d", &expected_edges) != 1 ||
      expected_edges <= 0) {
    fclose(fp);
    return NULL;
  }

  IntervalMap *intervals = interval_map_alloc(expected_edges);
  if (!intervals) {
    fclose(fp);
    return NULL;
  }

  int count = 0;
  while (fgets(line, sizeof(line), fp)) {
    int src_id, tgt_id;
    char start_str[64], end_str[64];

    int parsed = sscanf(line, " %d, %d, %63[^,], %63[^;];", &src_id, &tgt_id,
                        start_str, end_str);
    if (parsed != 4)
      continue;

    int64_t start = parse_iso8601_to_unix(start_str);
    int64_t end = parse_iso8601_to_unix(end_str);
    if (start == -1 || end == -1)
      continue;

    char src_buf[12], tgt_buf[12];
    snprintf(src_buf, sizeof(src_buf), "%d", src_id);
    snprintf(tgt_buf, sizeof(tgt_buf), "%d", tgt_id);

    int src_index = get_or_create_node_index(src_buf);
    int tgt_index = get_or_create_node_index(tgt_buf);

    intervals[count].start = start;
    intervals[count].end = end;
    intervals[count].target_id = pack_node_pair(src_index, tgt_index);
    intervals[count].sublist = -1;

    if (count < DEBUG_PRINT_COUNT) {
      printf("[DEBUG] Edge %d: %d -> %d | Start: %I64d | End: %I64d\n", count,
             src_id, tgt_id, start, end);
    }

    count++;
    if (count >= expected_edges)
      break;
  }

  fclose(fp);
  *p_count = count;

  printf("[INFO] Parsed %d temporal edges from '%s'\n", count, filename);

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
