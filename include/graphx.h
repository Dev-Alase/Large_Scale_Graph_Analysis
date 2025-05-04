#ifndef GRAPHX_H
#define GRAPHX_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <string>
#include <stdexcept>

namespace graphx {

// Type definitions for graph processing
using vertex_t = int;
using edge_t = int;
using weight_t = float;

// CUDA error checking utility
inline void checkCuda(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(func) + " failed: "
            + cudaGetErrorString(result) + " at "
            + file + ":" + std::to_string(line));
    }
}
#define CHECK_CUDA(x) checkCuda((x), #x, __FILE__, __LINE__)

// CSR Graph structure
struct CSRGraph {
    // Device pointers
    vertex_t* d_row_offsets;   // [num_vertices + 1]
    edge_t* d_col_indices;     // [num_edges]
    weight_t* d_values;        // [num_edges] (optional)
    
    // Host mirrors
    std::vector<vertex_t> h_row_offsets;
    std::vector<edge_t> h_col_indices;
    std::vector<weight_t> h_values;
    
    // Graph properties
    vertex_t num_vertices;
    edge_t num_edges;
    bool has_weights;

    CSRGraph() : d_row_offsets(nullptr), d_col_indices(nullptr), 
                 d_values(nullptr), num_vertices(0), num_edges(0),
                 has_weights(false) {}
    
    ~CSRGraph() { freeMemory(); }

    void freeMemory() {
        if (d_row_offsets) CHECK_CUDA(cudaFree(d_row_offsets));
        if (d_col_indices) CHECK_CUDA(cudaFree(d_col_indices));
        if (d_values) CHECK_CUDA(cudaFree(d_values));
        d_row_offsets = nullptr;
        d_col_indices = nullptr;
        d_values = nullptr;
    }
};

// Graph loading and generation
CSRGraph loadFromEdgeList(const std::string& filename, bool weighted = false);
CSRGraph generateRandomGraph(vertex_t num_vertices, float density, 
                           bool weighted = false, unsigned int seed = 42);

// Graph conversion
void convertToCSR(const std::vector<std::pair<vertex_t, vertex_t>>& edges,
                 const std::vector<weight_t>& weights,
                 vertex_t num_vertices,
                 CSRGraph& graph);

// PageRank configuration
struct PageRankConfig {
    float damping_factor = 0.85f;
    float tolerance = 1e-6f;
    int max_iterations = 100;
    int block_size = 256;
};

// PageRank kernel declarations
__global__ void initializePageRankKernel(float* d_ranks, 
                                       vertex_t num_vertices);

__global__ void pageRankIterationKernel(const vertex_t* d_row_offsets,
                                      const edge_t* d_col_indices,
                                      const float* d_old_ranks,
                                      float* d_new_ranks,
                                      float damping_factor,
                                      vertex_t num_vertices);

__global__ void computeErrorKernel(const float* d_old_ranks,
                                 const float* d_new_ranks,
                                 float* d_error,
                                 vertex_t num_vertices);

// PageRank host interface
void computePageRank(const CSRGraph& graph,
                    std::vector<float>& ranks,
                    const PageRankConfig& config = PageRankConfig());

// Memory management
void allocateGraphMemory(CSRGraph& graph);
void copyGraphToDevice(CSRGraph& graph);
void copyGraphToHost(CSRGraph& graph);

// Graph validation
bool validateGraph(const CSRGraph& graph);

} // namespace graphx

#endif // GRAPHX_H
