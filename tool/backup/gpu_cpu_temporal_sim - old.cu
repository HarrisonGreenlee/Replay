#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* for strcmp */
#include <time.h>

#define EIGEN_USE_THREADS
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cusparse_v2.h>

extern "C" {
#include "intervaldb.h" /* for IntervalIterator, find_intervals, etc. */
#include "temporal_contact_matrix.h" /* <-- Our TCM library header */
}

/* ----------------------------------------------------------------------------
 * Simulation parameters
 * ----------------------------------------------------------------------------*/

static int num_nodes;
static int gM = 1000;             // Default number of parallel simulations
static float gStepSize = 3600.0f; // Default step size
static int gIterations = 42;      // Default number of iterations
static int gEigenThreads = -1;    // parallel threads for Eigen, default to max

FILE *summary_fp = NULL;
FILE *node_fp = NULL;

// Disease-state boundaries
#define SUSCEPTIBLE 0
static int gUpperRange = 7200;
static int gMediumRange = 3600; // no resistance
static int gLowerRange = 0;
static float gInitialInfectedProb =
    0.5f; // Probability an individual starts infected
static float gSusceptibleInfectProb =
    0.99f; // chance to get infected if exposed

static int64_t gGlobalStartTime = 946713600; // Jan 1, 2000
static int64_t gTimeStep = 3600;             // 1 hour

// Simple CUDA check macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/* ----------------------------------------------------------------------------
 * Device kernels
 * ----------------------------------------------------------------------------*/

__global__ void initialize_countdown_vector(int *countdown_vector,
                                            int totalSize, float infected_prob,
                                            float step_size, unsigned long seed,
                                            int medium_range,
                                            int susceptible_value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < totalSize) {
    curandState state;
    curand_init(clock64() + idx + seed, 0, 0, &state);
    float rand_val = curand_uniform(&state);

    if (rand_val < infected_prob) {
      countdown_vector[idx] = medium_range + step_size;
    } else {
      countdown_vector[idx] = susceptible_value;
    }
  }
}

__global__ void generate_infectious_vector(const int *countdown_vector,
                                           float *infectious_vector,
                                           int n, // num_nodes
                                           int m, // gM
                                           int lower_range, int medium_range) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * m;
  if (idx < total) {
    // Convert idx -> (row, col) in col-major
    int row = idx % n;
    int col = idx / n;

    int c = countdown_vector[row + col * n];
    if (c > lower_range && c <= medium_range) {
      infectious_vector[row + col * n] = 1.0f;
    } else {
      infectious_vector[row + col * n] = 0.0f;
    }
  }
}

__global__ void compute_infection_probability(float *exposure_matrix,
                                              const int *countdown_vector,
                                              int n, int m, float step_size,
                                              float infect_prob,
                                              int susceptible_value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * m;
  if (idx < total) {
    int row = idx % n;
    int col = idx / n;

    if (countdown_vector[row + col * n] == susceptible_value) {
      float log_base = logf(1.0f - infect_prob);
      //  we do not normalize wrt step size anymore - we now normalize wrt a
      //  standard "exposure hour". Just easier to work with. float e =
      //  exposure_matrix[row + col*n] / step_size;
      float e = exposure_matrix[row + col * n] / 3600.0f;
      exposure_matrix[row + col * n] = 1.0f - expf(log_base * e);
    } else {
      exposure_matrix[row + col * n] = 0.0f;
    }
  }
}

__global__ void monte_carlo_simulation(const float *prob_matrix,
                                       int *countdown_vector, int n, int m,
                                       unsigned long seed, float step_size,
                                       int upper_range, int susceptible_value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * m;
  if (idx < total) {
    int row = idx % n;
    int col = idx / n;

    if (countdown_vector[row + col * n] == susceptible_value) {
      curandState state;
      curand_init(clock64() + idx + seed, 0, 0, &state);
      float rand_val = curand_uniform(&state);

      float infectionProb = prob_matrix[row + col * n];
      if (infectionProb > rand_val) {
        // New infection => set to incubation
        countdown_vector[row + col * n] = upper_range + step_size;
      }
    }
  }
}

__global__ void update_countdown_vector(int *countdown_vector, int n, int m,
                                        float step_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * m;
  if (idx < total) {
    int row = idx % n;
    int col = idx / n;

    if (countdown_vector[row + col * n] > 0) {
      countdown_vector[row + col * n] -= step_size;
    }
    if (countdown_vector[row + col * n] < 0) {
      countdown_vector[row + col * n] = SUSCEPTIBLE;
    }
  }
}

/* ----------------------------------------------------------------------------
 * Utility to print first few "rows" (meaning col=0..X in col-major)
 * ----------------------------------------------------------------------------*/
void print_status_counts(const int *h_countdown_vector, int n, int m,
                         int rowsToPrint) {
  for (int i = 0; i < rowsToPrint; i++) {
    int incubating = 0, infectious = 0, resistant = 0, susceptible = 0;
    for (int j = 0; j < n; j++) {
      int offset = j + i * n; // row=j, col=i
      int status = h_countdown_vector[offset];

      if (status == SUSCEPTIBLE)
        susceptible++;
      else if (status <= gLowerRange && status > 0)
        resistant++;
      else if (status <= gMediumRange)
        infectious++;
      else if (status > gMediumRange)
        incubating++;
    }
    printf("Row %d: Exposed=%d, Infectious=%d, Resistant=%d, Susceptible=%d\n",
           i, incubating, infectious, resistant, susceptible);
  }
}

/* ----------------------------------------------------------------------------
 * Build a CSR adjacency from the intervals that overlap [start_ts, end_ts)
 * We collect edges in a simple array, then sort them by row -> col, then
 * populate the CSR arrays.
 * ----------------------------------------------------------------------------*/
typedef struct {
  int row;
  int col;
  float weight;
} EdgeListItem;

/* Compare function for qsort (by row, then col) */
static int cmp_edgelist(const void *a, const void *b) {
  const EdgeListItem *ea = (const EdgeListItem *)a;
  const EdgeListItem *eb = (const EdgeListItem *)b;
  if (ea->row < eb->row)
    return -1;
  if (ea->row > eb->row)
    return 1;
  /* same row => compare col */
  if (ea->col < eb->col)
    return -1;
  if (ea->col > eb->col)
    return 1;
  return 0;
}

/*
 * build_csr_from_intervals:
 *   - 'hits' is the list of intervals that overlap [start_ts, end_ts).
 *   - 'n_hits' is how many we have.
 *   - If we treat them as *undirected*, we add both (src, tgt) and (tgt, src).
 *   - 'row' = src, 'col' = tgt, weight = overlap_duration.
 */
static void
build_csr_from_intervals(IntervalMap *hits, int n_hits,
                         int n_nodes, /* = number of distinct nodes => replaced
                                         usage of N with num_nodes */
                         int **csrRowPtr, int **csrColInd, float **csrVal,
                         size_t *p_nnz, int64_t start_ts, int64_t end_ts) {
  /* We'll store edges in a dynamic array.
     In the worst case for an undirected approach, we might have up to 2*n_hits.
   */
  EdgeListItem *edgelist =
      (EdgeListItem *)malloc(sizeof(EdgeListItem) * (size_t)2 * n_hits);
  if (!edgelist) {
    fprintf(stderr, "Out of memory building edgelist.\n");
    exit(1);
  }
  int ecount = 0;

  /* For each interval in hits, compute actual overlap, then add edge(s). */
  for (int i = 0; i < n_hits; i++) {
    int64_t s = hits[i].start;
    int64_t e = hits[i].end;
    int src_idx, tgt_idx;
    unpack_node_pair(hits[i].target_id, &src_idx, &tgt_idx);

    int64_t overlap_start = (s > start_ts) ? s : start_ts;
    int64_t overlap_end = (e < end_ts) ? e : end_ts;
    if (overlap_end > overlap_start) {
      float overlap_duration = (float)(overlap_end - overlap_start);

      /* Add both directions if you want an undirected adjacency: */
      edgelist[ecount].row = src_idx;
      edgelist[ecount].col = tgt_idx;
      edgelist[ecount].weight = overlap_duration;
      ecount++;

      edgelist[ecount].row = tgt_idx;
      edgelist[ecount].col = src_idx;
      edgelist[ecount].weight = overlap_duration;
      ecount++;
    }
  }

  /* Sort the edges by (row, col) */
  qsort(edgelist, ecount, sizeof(EdgeListItem), cmp_edgelist);

  /* Now build CSR from the sorted edge list.
     1) We know n_nodes = N (the dimension).
     2) The # of nonzeros = ecount. */

  *p_nnz = (size_t)ecount;
  *csrRowPtr = (int *)malloc((size_t)(n_nodes + 1) * sizeof(int));
  *csrColInd = (int *)malloc((size_t)(ecount) * sizeof(int));
  *csrVal = (float *)malloc((size_t)(ecount) * sizeof(float));

  if (!(*csrRowPtr) || !(*csrColInd) || !(*csrVal)) {
    fprintf(stderr, "Out of memory allocating CSR arrays.\n");
    free(edgelist);
    exit(1);
  }

  /* Initialize rowPtr to 0. We'll accumulate counts. */
  for (int i = 0; i <= n_nodes; i++) {
    (*csrRowPtr)[i] = 0;
  }

  /* Count how many entries in each row. */
  for (int i = 0; i < ecount; i++) {
    int r = edgelist[i].row;
    (*csrRowPtr)[r + 1] += 1;
  }

  /* Convert to prefix sum for rowPtr. */
  for (int i = 0; i < n_nodes; i++) {
    (*csrRowPtr)[i + 1] += (*csrRowPtr)[i];
  }

  /* Now fill colInd/Val. We'll keep a "current index" for each row. */
  int *rowPosition = (int *)malloc((size_t)n_nodes * sizeof(int));
  for (int i = 0; i < n_nodes; i++) {
    rowPosition[i] = (*csrRowPtr)[i];
  }

  /* Place edges into correct position. */
  for (int i = 0; i < ecount; i++) {
    int r = edgelist[i].row;
    int pos = rowPosition[r];
    (*csrColInd)[pos] = edgelist[i].col;
    (*csrVal)[pos] = edgelist[i].weight;
    rowPosition[r]++;
  }
  free(rowPosition);
  free(edgelist);
}

/* ----------------------------------------------------------------------------
 * CPU FALLBACK FUNCTIONS (newly added):
 * We replicate the logic of the CUDA kernels and SpMM using the host CPU.
 * These are only called if --cpu-only is specified.
 * ----------------------------------------------------------------------------*/

/*
 * CPUInitializeCountdownVector
 *  - Replicates initialize_countdown_vector kernel
 */
void CPUInitializeCountdownVector(int *countdown_vector, int totalSize,
                                  float infected_prob, float step_size) {
  // Simple seed using time(NULL), but we do it outside to keep it consistent
  srand((unsigned)time(NULL));
  for (int idx = 0; idx < totalSize; idx++) {
    float rand_val = (float)rand() / (float)RAND_MAX;
    if (rand_val < infected_prob) {
      countdown_vector[idx] = gMediumRange + step_size;
    } else {
      countdown_vector[idx] = SUSCEPTIBLE;
    }
  }
}

/*
 * CPUGenerateInfectiousVector
 *   - Replicates generate_infectious_vector kernel
 *   - NxM stored in column-major => element at [row + col*n].
 */
void CPUGenerateInfectiousVector(const int *countdown_vector,
                                 float *infectious_vector, int n, int m) {
  int total = n * m;
  for (int idx = 0; idx < total; idx++) {
    int row = idx % n;
    int col = idx / n;
    int c = countdown_vector[row + col * n];
    if (c > gLowerRange && c <= gMediumRange) {
      infectious_vector[idx] = 1.0f;
    } else {
      infectious_vector[idx] = 0.0f;
    }
  }
}

/*
 * CPUComputeSpMM
 *   - Replicates A * B => C, where A is CSR of size NxN,
 *     B is NxM (col-major),
 *     C is NxM (col-major).
 *   - alpha=1.0f, beta=0.0f
 */
void CPUComputeSpMM(const int *csrRowPtr, const int *csrColInd,
                    const float *csrVal, size_t nnz_count,
                    int n,                          // NxN
                    const float *infectious_vector, // NxM
                    float *exposure_matrix,         // NxM
                    int m) {
  using namespace Eigen;

  // Step 1: Map your CSR arrays into an Eigen::SparseMatrix
  typedef Eigen::SparseMatrix<float, Eigen::RowMajor, int> SpMat;
  SpMat A(n, n);

  std::vector<Triplet<float>> triplets;
  triplets.reserve(nnz_count);
  for (size_t i = 0; i < (size_t)n; i++) {
    for (int j = csrRowPtr[i]; j < csrRowPtr[i + 1]; j++) {
      triplets.push_back(Triplet<float>(i, csrColInd[j], csrVal[j]));
    }
  }
  A.setFromTriplets(triplets.begin(), triplets.end());

  // Step 2: Map infectious_vector as a dense matrix (column-major!)
  Map<const Matrix<float, Dynamic, Dynamic, ColMajor>> B(infectious_vector, n,
                                                         m);

  // Step 3: Map exposure_matrix as output
  Map<Matrix<float, Dynamic, Dynamic, ColMajor>> C(exposure_matrix, n, m);

  // Step 4: Sparse × Dense multiplication
  C = A * B;
}

/*
 * CPUComputeInfectionProbability
 *   - Replicates compute_infection_probability kernel
 */
void CPUComputeInfectionProbability(float *exposure_matrix,
                                    const int *countdown_vector, int n, int m,
                                    float step_size) {
  int total = n * m;
  for (int idx = 0; idx < total; idx++) {
    int row = idx % n;
    int col = idx / n;
    if (countdown_vector[row + col * n] == SUSCEPTIBLE) {
      float infect_prob = gSusceptibleInfectProb;
      float log_base = logf(1.0f - infect_prob);
      //  float e = exposure_matrix[idx] / step_size;
      //  we do not normalize wrt step size anymore - we now normalize wrt a
      //  standard "exposure hour". Just easier to work with.
      float e = exposure_matrix[row + col * n] / 3600.0f;
      exposure_matrix[idx] = 1.0f - expf(log_base * e);
    } else {
      exposure_matrix[idx] = 0.0f;
    }
  }
}

/*
 * CPUMonteCarloSimulation
 *   - Replicates monte_carlo_simulation kernel
 */
void CPUMonteCarloSimulation(const float *prob_matrix, int *countdown_vector,
                             int n, int m, float step_size) {
  // We'll do a simple rand() approach again.
  // You can seed it once per iteration for demonstration.
  srand((unsigned)time(NULL));
  int total = n * m;
  for (int idx = 0; idx < total; idx++) {
    int row = idx % n;
    int col = idx / n;

    if (countdown_vector[row + col * n] == SUSCEPTIBLE) {
      float rand_val = (float)rand() / (float)RAND_MAX;
      float infectionProb = prob_matrix[idx];
      if (infectionProb > rand_val) {
        // New infection => set to incubation
        countdown_vector[row + col * n] = gUpperRange + step_size;
      }
    }
  }
}

/*
 * CPUUpdateCountdownVector
 *   - Replicates update_countdown_vector kernel
 */
void CPUUpdateCountdownVector(int *countdown_vector, int n, int m,
                              float step_size) {
  int total = n * m;
  for (int idx = 0; idx < total; idx++) {
    if (countdown_vector[idx] > 0) {
      countdown_vector[idx] -= step_size;
    }
    if (countdown_vector[idx] < 0) {
      countdown_vector[idx] = SUSCEPTIBLE;
    }
  }
}
/* ----------------------------------------------------------------------------
 * END of CPU FALLBACK FUNCTIONS
 * ----------------------------------------------------------------------------*/

void write_simulation_state(FILE *summary_fp, FILE *node_fp,
                            const int *countdown_vector, int num_nodes, int gM,
                            int64_t timestamp) {
  for (int sim = 0; sim < gM; sim++) {
    int incubating = 0, infectious = 0, resistant = 0, susceptible = 0;

    for (int node = 0; node < num_nodes; node++) {
      int idx = node + sim * num_nodes;
      int state_val = countdown_vector[idx];

      const char *state_str;
      if (state_val == SUSCEPTIBLE) {
        susceptible++;
        state_str = "susceptible";
      } else if (state_val > gMediumRange) {
        incubating++;
        state_str = "incubating";
      } else if (state_val > gLowerRange) {
        infectious++;
        state_str = "infectious";
      } else {
        resistant++;
        state_str = "resistant";
      }

      if (node_fp) {
        fprintf(node_fp, "%lld,%d,%d,%s\n", (long long)timestamp, sim, node,
                state_str);
      }
    }

    if (summary_fp) {
      fprintf(summary_fp, "%lld,%d,%d,%d,%d,%d\n", (long long)timestamp, sim,
              incubating, infectious, resistant, susceptible);
    }
  }

  if (summary_fp)
    fflush(summary_fp);
  if (node_fp)
    fflush(node_fp);
}

int main(int argc, char **argv) {
  printf("=== GPU-Accelerated Temporal Epidemiology Simulation ===\n");

  if (argc < 2) {
    fprintf(stderr, "Usage: %s <graph file> [options]\n", argv[0]);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --cpu-only             Run on CPU only.\n");
    fprintf(
        stderr,
        "  --cpu-threads <int>    Number of CPU threads Eigen should use.\n");
    fprintf(stderr,
            "  --N <int>              Number of individuals per simulation.\n");
    fprintf(stderr,
            "  --M <int>              Number of parallel simulations.\n");
    fprintf(stderr, "  --step-size <float>    Time step size.\n");
    fprintf(stderr,
            "  --iterations <int>     Number of main simulation steps.\n");
    fprintf(stderr,
            "  --summary-out <file>   Output file for summary state counts.\n");
    fprintf(stderr,
            "  --node-state-out <file> Output file for per-node states.\n");
    fprintf(stderr, "  --initial-infected <float>     Initial infection "
                    "probability [0–1]\n");
    fprintf(
        stderr,
        "  --infect-prob <float>          Infection prob if exposed [0–1]\n");
    fprintf(stderr, "  --upper-range <int>            Countdown time for "
                    "resistant phase\n");
    fprintf(stderr, "  --medium-range <int>           Countdown time for "
                    "infectious phase\n");
    fprintf(
        stderr,
        "  --lower-range <int>            Threshold for susceptible state\n");
    fprintf(
        stderr,
        "  --start-time <epoch>         Unix start time of the simulation.\n");
    fprintf(stderr, "  --time-step <seconds>        Duration of each "
                    "simulation step in seconds.\n");

    return 1;
  }

  const char *filename = NULL;
  bool cpu_only = false;

  // Simple argument parsing:
  for (int i = 1; i < argc; i++) {
    if (argv[i][0] != '-') {
      filename = argv[i];
      continue;
    }
    if (strcmp(argv[i], "--cpu-only") == 0) {
      cpu_only = true;
    } else if (strcmp(argv[i], "--cpu-threads") == 0 && (i + 1 < argc)) {
      gEigenThreads = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--M") == 0 && (i + 1 < argc)) {
      gM = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--step-size") == 0 && (i + 1 < argc)) {
      gStepSize = (float)atof(argv[++i]);
    } else if (strcmp(argv[i], "--iterations") == 0 && (i + 1 < argc)) {
      gIterations = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--summary-out") == 0 && (i + 1 < argc)) {
      remove(argv[i + 1]); // delete existing file if present
      summary_fp = fopen(argv[++i], "w");
      if (!summary_fp) {
        fprintf(stderr, "Error: Could not open summary output file.\n");
        return 1;
      }
    } else if (strcmp(argv[i], "--node-state-out") == 0 && (i + 1 < argc)) {
      remove(argv[i + 1]); // delete existing file if present
      node_fp = fopen(argv[++i], "w");
      if (!node_fp) {
        fprintf(stderr, "Error: Could not open node state output file.\n");
        return 1;
      }
    } else if (strcmp(argv[i], "--initial-infected") == 0 && (i + 1 < argc)) {
      gInitialInfectedProb = (float)atof(argv[++i]);
    } else if (strcmp(argv[i], "--infect-prob") == 0 && (i + 1 < argc)) {
      gSusceptibleInfectProb = (float)atof(argv[++i]);
    } else if (strcmp(argv[i], "--upper-range") == 0 && (i + 1 < argc)) {
      gUpperRange = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--medium-range") == 0 && (i + 1 < argc)) {
      gMediumRange = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--lower-range") == 0 && (i + 1 < argc)) {
      gLowerRange = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--start-time") == 0 && (i + 1 < argc)) {
      gGlobalStartTime = (int64_t)atoll(argv[++i]);
    } else if (strcmp(argv[i], "--time-step") == 0 && (i + 1 < argc)) {
      gTimeStep = (int64_t)atoll(argv[++i]);
    }

    else {
      fprintf(stderr, "Unrecognized option: %s\n", argv[i]);
      return 1;
    }
  }

  if (!filename) {
    fprintf(stderr, "Missing contact network file argument.\n");
    return 1;
  }

  // After file opening, write headers:
  if (summary_fp) {
    fprintf(summary_fp,
            "time,simulation_id,exposed,infectious,resistant,susceptible\n");
  }
  if (node_fp) {
    fprintf(node_fp, "time,simulation_id,node_id,state\n");
  }

  if (cpu_only) {
    if (gEigenThreads > 0) {
      Eigen::setNbThreads(gEigenThreads);
      printf("[INFO] Eigen thread count set to %d\n", Eigen::nbThreads());
    } else {
      // Let Eigen decide
      printf("[INFO] Eigen will auto-select thread count: %d\n",
             Eigen::nbThreads());
    }
  }

  /* 1) Parse graph => intervals */
  int edge_count = 0;
  IntervalMap *intervals =
      parse_edgelist_build_intervals(filename, &edge_count);
  if (!intervals || edge_count == 0) {
    fprintf(stderr, "No intervals read or parse error.\n");
    free_node_map();
    return 1;
  }
  fprintf(stderr, "Parsed %d edges (intervals) from graph.\n", edge_count);
  num_nodes = get_total_node_count();
  printf("[INFO] Node count set from NODE_LIST= line: %d\n", num_nodes);

  /* 2) Build NCLS from intervals */
  IntervalDBWrapper dbw;
  if (!build_interval_db_wrapper(intervals, edge_count, &dbw)) {
    fprintf(stderr, "Error: Failed to build interval DB.\n");
    free_node_map();
    return 1;
  }
  fprintf(stderr, "NCLS built. n=%d, nlists=%d.\n", dbw.n, dbw.nlists);

  /* 3) Simulation Setup */
  size_t totalSize = (size_t)num_nodes * (size_t)gM;

  // We will keep host arrays for CPU or for debugging GPU results
  int *h_countdown_vector = NULL;
  float *h_exposure_matrix = NULL;
  float *h_infectious_vector = NULL;

  // Allocate host arrays
  // For CPU mode, we do everything in these arrays
  // For GPU mode, we also store the data on GPU, but keep these for debug
  h_countdown_vector = (int *)malloc(totalSize * sizeof(int));
  h_exposure_matrix = (float *)malloc(totalSize * sizeof(float));
  h_infectious_vector = (float *)malloc(totalSize * sizeof(float));
  if (!h_countdown_vector || !h_exposure_matrix || !h_infectious_vector) {
    fprintf(stderr, "Error: unable to allocate host arrays.\n");
    free_node_map();
    free_interval_db_wrapper(&dbw);
    return 1;
  }

  // If not CPU-only, allocate GPU buffers:
  int *d_countdown_vector = NULL;
  float *d_exposure_matrix = NULL;
  float *d_infectious_vector = NULL;

  if (!cpu_only) {
    CUDA_CHECK(
        cudaMalloc((void **)&d_countdown_vector, totalSize * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc((void **)&d_exposure_matrix, totalSize * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc((void **)&d_infectious_vector, totalSize * sizeof(float)));
  }

  // Initialize countdown vector
  if (cpu_only) {
    CPUInitializeCountdownVector(h_countdown_vector, (int)totalSize,
                                 gInitialInfectedProb, gStepSize);
  } else {
    // GPU initialization
    int threads = 256;
    int blocks = (int)((totalSize + threads - 1) / threads);
    initialize_countdown_vector<<<blocks, threads>>>(
        d_countdown_vector, (int)totalSize, gInitialInfectedProb, gStepSize,
        (unsigned long)time(NULL), gMediumRange, SUSCEPTIBLE);
    CUDA_CHECK(cudaDeviceSynchronize());
    // Copy to host for the initial debug
    CUDA_CHECK(cudaMemcpy(h_countdown_vector, d_countdown_vector,
                          totalSize * sizeof(int), cudaMemcpyDeviceToHost));
  }

  // Create cuSPARSE handle (only if not CPU-only)
  cusparseHandle_t cusparseHandle = NULL;
  if (!cpu_only) {
    cusparseCreate(&cusparseHandle);
  }

  // Prepare DnMat descriptors (only if not CPU-only)
  cusparseDnMatDescr_t matB = NULL, matC = NULL;
  if (!cpu_only) {
    cusparseCreateDnMat(&matB, (int64_t)num_nodes, (int64_t)gM,
                        (int64_t)num_nodes, d_infectious_vector, CUDA_R_32F,
                        CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&matC, (int64_t)num_nodes, (int64_t)gM,
                        (int64_t)num_nodes, d_exposure_matrix, CUDA_R_32F,
                        CUSPARSE_ORDER_COL);
  }

  float alpha = 1.0f, beta = 0.0f;
  void *dBuffer = NULL;
  size_t bufferSize = 0;

  cusparseSpMatDescr_t matA = NULL; // adjacency

  // write initial conditions to file
  if (summary_fp || node_fp) { // if file output flags exist
    write_simulation_state(summary_fp, node_fp, h_countdown_vector, num_nodes,
                           gM, gGlobalStartTime);
  }

  printf("Starting temporal Monte Carlo simulation for %d iterations...\n",
         gIterations);
  printf("[INFO] Simulation start time: %lld, time step: %lld seconds\n",
         (long long)gGlobalStartTime, (long long)gTimeStep);
  printf("[DEBUG] num_nodes = %d\n", num_nodes);

  // Basic GPU thread config for kernels (used if not CPU-only)
  int threads = 256;
  int blocks = (int)((totalSize + threads - 1) / threads);

  /* 4) Main iteration loop */
  for (int iter = 0; iter < gIterations; iter++) {
    int64_t offset = (int64_t)iter * gTimeStep;
    if ((gTimeStep > 0 && iter > 0 && offset < 0) ||
        (gTimeStep < 0 && iter > 0 && offset > 0)) {
      fprintf(stderr, "Error: Overflow detected when computing offset.\n");
      return 1;
    }

    int64_t current_start_ts = gGlobalStartTime + offset;
    if ((gTimeStep > 0 && current_start_ts > INT64_MAX - gTimeStep) ||
        (gTimeStep < 0 && current_start_ts < INT64_MIN - gTimeStep)) {
      fprintf(stderr,
              "Error: Overflow detected when computing current_end_ts.\n");
      return 1;
    }

    int64_t current_end_ts = current_start_ts + gTimeStep;

    printf("\n--- Iteration %d: time window [%lld, %lld) ---\n", iter + 1,
           (long long)current_start_ts, (long long)current_end_ts);

    /*
     * Allocate buffer for overlapping intervals ("hits").
     * Normally, we use dbw.nlists (number of sublists) as the max expected
     * overlap count. However, if all intervals are top-level (i.e., no
     * nesting), dbw.nlists == 0, and using it would result in a zero-sized
     * buffer. In that case, we conservatively allocate up to dbw.n (total
     * number of intervals), which guarantees enough space.
     */
    IntervalMap *hits = (IntervalMap *)malloc(
        sizeof(IntervalMap) * (dbw.nlists > 0 ? dbw.nlists : dbw.n));

    if (!hits) {
      fprintf(stderr, "Out of memory for hits.\n");
      exit(1);
    }
    IntervalIterator *it = interval_iterator_alloc();
    if (!it) {
      fprintf(stderr, "Out of memory for IntervalIterator.\n");
      free(hits);
      exit(1);
    }
    int n_return = 0;

    find_intervals(it, current_start_ts, current_end_ts, dbw.im, dbw.n,
                   dbw.subheader, dbw.nlists, hits, dbw.nlists, &n_return, &it);

    free_interval_iterator(it);

    /* 4.2) Convert the returned intervals to CSR adjacency. */
    int *h_csrRowPtr = NULL;
    int *h_csrColInd = NULL;
    float *h_csrVal = NULL;
    size_t nnz_count = 0;

    build_csr_from_intervals(hits, n_return, num_nodes, &h_csrRowPtr,
                             &h_csrColInd, &h_csrVal, &nnz_count,
                             current_start_ts, current_end_ts);

    free(hits);

    printf("  iteration %d => found %d intervals => nnz=%zu\n", iter + 1,
           n_return, nnz_count);

    // If CPU-only, we won't copy to device; we'll do a CPU spMM
    int *d_csrRowPtr = NULL;
    int *d_csrColInd = NULL;
    float *d_csrVal = NULL;

    if (!cpu_only) {
      /* 4.3) Copy CSR to GPU memory. */
      CUDA_CHECK(
          cudaMalloc((void **)&d_csrRowPtr, (num_nodes + 1) * sizeof(int)));
      CUDA_CHECK(cudaMalloc((void **)&d_csrColInd, nnz_count * sizeof(int)));
      CUDA_CHECK(cudaMalloc((void **)&d_csrVal, nnz_count * sizeof(float)));

      CUDA_CHECK(cudaMemcpy(d_csrRowPtr, h_csrRowPtr,
                            (num_nodes + 1) * sizeof(int),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_csrColInd, h_csrColInd, nnz_count * sizeof(int),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_csrVal, h_csrVal, nnz_count * sizeof(float),
                            cudaMemcpyHostToDevice));

      /* 4.4) Create/Update the cuSPARSE SpMat descriptor for A. */
      if (matA) {
        cusparseDestroySpMat(matA);
        matA = NULL;
      }
      cusparseCreateCsr(&matA, (int64_t)num_nodes, (int64_t)num_nodes,
                        (int64_t)nnz_count, d_csrRowPtr, d_csrColInd, d_csrVal,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

      /* 4.5) We now must get buffer size for SpMM with this adjacency. */
      cusparseSpMM_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                              matB, &beta, matC, CUDA_R_32F,
                              CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);

      if (dBuffer) {
        cudaFree(dBuffer);
        dBuffer = NULL;
      }
      CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));
    }

    /* 4.6) Steps: Infectious vector, SpMM => exposure, infection prob, etc. */

    if (!cpu_only) {
      // GPU-based approach
      // 4.6a: generate_infectious_vector
      generate_infectious_vector<<<blocks, threads>>>(
          d_countdown_vector, d_infectious_vector, num_nodes, gM, gLowerRange,
          gMediumRange);
      CUDA_CHECK(cudaDeviceSynchronize());

      // 4.6b: SpMM => exposure_matrix = A * infectious_vector
      cusparseSpMM(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                   CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta,
                   matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
      CUDA_CHECK(cudaDeviceSynchronize());

      // 4.6c: compute_infection_probability
      compute_infection_probability<<<blocks, threads>>>(
          d_exposure_matrix, d_countdown_vector, num_nodes, gM, gStepSize,
          gSusceptibleInfectProb, SUSCEPTIBLE);
      CUDA_CHECK(cudaDeviceSynchronize());

      // 4.6d: monte_carlo_simulation
      monte_carlo_simulation<<<blocks, threads>>>(
          d_exposure_matrix, d_countdown_vector, num_nodes, gM,
          (unsigned long)(time(NULL) + iter), gStepSize, gUpperRange,
          SUSCEPTIBLE);

      CUDA_CHECK(cudaDeviceSynchronize());

      // 4.6e: update_countdown_vector
      update_countdown_vector<<<blocks, threads>>>(d_countdown_vector,
                                                   num_nodes, gM, gStepSize);
      CUDA_CHECK(cudaDeviceSynchronize());

      // For debug printing
      CUDA_CHECK(cudaMemcpy(h_countdown_vector, d_countdown_vector,
                            totalSize * sizeof(int), cudaMemcpyDeviceToHost));
      print_status_counts(h_countdown_vector, num_nodes, gM, 5);

      if (summary_fp || node_fp) { // if file output flags set
        int64_t current_time = gGlobalStartTime + (int64_t)iter * gTimeStep;
        write_simulation_state(summary_fp, node_fp, h_countdown_vector,
                               num_nodes, gM, current_time);
      }

      /* 4.7) Cleanup adjacency from GPU */
      if (matA) {
        cusparseDestroySpMat(matA);
        matA = NULL;
      }
      CUDA_CHECK(cudaFree(d_csrRowPtr));
      CUDA_CHECK(cudaFree(d_csrColInd));
      CUDA_CHECK(cudaFree(d_csrVal));
      d_csrRowPtr = NULL;
      d_csrColInd = NULL;
      d_csrVal = NULL;
    } else {
      // CPU-only approach
      // 4.6a: generate_infectious_vector
      CPUGenerateInfectiousVector(h_countdown_vector, h_infectious_vector,
                                  num_nodes, gM);

      // 4.6b: SpMM => exposure_matrix = A * infectious_vector
      // (we have adjacency in h_csrRowPtr, h_csrColInd, h_csrVal)
      CPUComputeSpMM(h_csrRowPtr, h_csrColInd, h_csrVal, nnz_count, num_nodes,
                     h_infectious_vector, // NxM
                     h_exposure_matrix,   // NxM
                     gM);

      // 4.6c: compute_infection_probability
      CPUComputeInfectionProbability(h_exposure_matrix, h_countdown_vector,
                                     num_nodes, gM, gStepSize);

      // 4.6d: monte_carlo_simulation
      CPUMonteCarloSimulation(h_exposure_matrix, h_countdown_vector, num_nodes,
                              gM, gStepSize);

      // 4.6e: update_countdown_vector
      CPUUpdateCountdownVector(h_countdown_vector, num_nodes, gM, gStepSize);

      // Debug print first 5 "rows"
      print_status_counts(h_countdown_vector, num_nodes, gM, 5);

      if (summary_fp || node_fp) {
        int64_t current_time = gGlobalStartTime + (int64_t)iter * gTimeStep;
        write_simulation_state(summary_fp, node_fp, h_countdown_vector,
                               num_nodes, gM, current_time);
      }

      // CPU adjacency arrays can just be freed
    }

    // 4.7) Cleanup adjacency from CPU side
    free(h_csrRowPtr);
    free(h_csrColInd);
    free(h_csrVal);
  }

  printf("Simulation done.\n");

  /* 5) Cleanup GPU objects if used */
  if (!cpu_only) {
    if (dBuffer)
      cudaFree(dBuffer);
    if (matA)
      cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(cusparseHandle);

    CUDA_CHECK(cudaFree(d_countdown_vector));
    CUDA_CHECK(cudaFree(d_exposure_matrix));
    CUDA_CHECK(cudaFree(d_infectious_vector));
  }

  // Close files
  if (summary_fp)
    fclose(summary_fp);
  if (node_fp)
    fclose(node_fp);

  // Free host arrays
  free(h_countdown_vector);
  free(h_exposure_matrix);
  free(h_infectious_vector);

  /* 6) Cleanup NCLS + node map */
  free_interval_db_wrapper(&dbw);
  free_node_map();

  return 0;
}
