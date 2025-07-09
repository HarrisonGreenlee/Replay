/******************************************************************************
 * temporal_contact_matrix.h
 *
 * Public header exposing the data structures and functions from
 * "temporal_contact_matrix.c" for use by other modules.
 *
 * Usage:
 *   #include "temporal_contact_matrix.h"
 *   Link with temporal_contact_matrix.c and intervaldb.c (plus libxml2).
 ******************************************************************************/

#ifndef TEMPORAL_CONTACT_MATRIX_H
#define TEMPORAL_CONTACT_MATRIX_H

#include <stdint.h>

#include "uthash.h"

#include "intervaldb.h" /* for IntervalMap, SublistHeader, etc. */

/* --------------------------------------------------------------------------
 * Data structures from the original temporal_contact_matrix.c
 * -------------------------------------------------------------------------- */

/* The node map is used internally in temporal_contact_matrix.c. */
typedef struct {
  char *node_id;     /* original string from graph */
  int index;         /* compressed integer index */
  UT_hash_handle hh; /* uthash handle for fast lookup */
} NodeItem;

/* The IntervalMap is defined in intervaldb.h, but typically like:
 *
 * typedef struct IntervalMap {
 *     int64_t start;
 *     int64_t end;
 *     int64_t target_id;
 *     int     sublist;
 * } IntervalMap;
 *
 * We do not redefine it if intervaldb.h does so. Just be sure the types match.
 */

/* For storing the final nested list metadata */
typedef struct {
  IntervalMap *im;
  SublistHeader *subheader;
  int n;      /* number of intervals after compression */
  int nlists; /* number of sublists */
} IntervalDBWrapper;

/* --------------------------------------------------------------------------
 * Public function declarations
 * -------------------------------------------------------------------------- */

/* pack_node_pair / unpack_node_pair
 *   Combine or split (src, tgt) into a single 64-bit integer.
 */
int64_t pack_node_pair(int src, int tgt);
void unpack_node_pair(int64_t packed, int *src, int *tgt);

/* parse_edgelist_build_intervals:
 *   Reads the new NODE_LIST format and returns an IntervalMap array.
 *   Caller must free the returned array.
 */
IntervalMap *parse_edgelist_build_intervals(const char *filename, int *p_count);

/* Return the total number of nodes specified by NODE_LIST= */
int get_total_node_count(void);

/* Count valid edge lines (helper used internally) */
int count_valid_edges(const char *filename);

/* build_interval_db_wrapper:
 *   Takes the intervals, builds the nested list DB.
 *   Returns 1 on success, 0 on failure.
 */
int build_interval_db_wrapper(IntervalMap *im, int n, IntervalDBWrapper *dbw);

/* free_interval_db_wrapper:
 *   Cleans up resources in the wrapper.
 */
void free_interval_db_wrapper(IntervalDBWrapper *dbw);

/* free_node_map:
 *   Frees the hash-based node map used internally for ID -> index compression.
 */
void free_node_map(void);

/* Convert ISO8601 string to Unix timestamp (UTC). */
int64_t parse_iso8601_to_unix(const char *time_str);

/* Return a unique compressed index for a node string (or create one). */
int get_or_create_node_index(const char *node_id_str);

#endif /* TEMPORAL_CONTACT_MATRIX_H */
