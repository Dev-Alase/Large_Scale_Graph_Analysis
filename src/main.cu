#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <string>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <numeric>  // For std::accumulate
#include <random>   // For random number generation
#include "../include/graphx.h"

// Timer utility for performance measurement
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    bool running;

public:
    Timer() : running(false) {}

    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        running = true;
    }

    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
        running = false;
    }

    double elapsedMilliseconds() {
        if (running) {
            auto now = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(now - start_time).count();
        }
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
};

namespace graphx {

// Implementation of PageRank kernels

// Initialize PageRank values and calculate outgoing edge counts
__global__ void initializePageRankKernel(float* d_ranks, vertex_t num_vertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vertices) {
        // Initialize with uniform probability 1/N
        d_ranks[tid] = 1.0f / num_vertices;
    }
}

// Compute outgoing contribution for each vertex
__global__ void computeContributionsKernel(
    const vertex_t* d_row_offsets,
    const vertex_t* d_degrees,
    const float* d_ranks,
    float* d_contributions,
    vertex_t num_vertices) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vertices) {
        int out_degree = d_degrees[tid];
        if (out_degree > 0) {
            // Each outgoing edge gets an equal share of this vertex's PageRank
            d_contributions[tid] = d_ranks[tid] / out_degree;
        } else {
            // Dangling node - contribution is distributed evenly later
            d_contributions[tid] = 0.0f;
        }
    }
}

// PageRank iteration kernel
__global__ void pageRankIterationKernel(
    const vertex_t* d_row_offsets,
    const edge_t* d_col_indices,
    const float* d_contributions,
    float* d_new_ranks,
    float damping_factor,
    float dangling_contribution,
    vertex_t num_vertices) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vertices) {
        float sum = 0.0f;
        
        // Sum contributions from incoming edges
        int start = d_row_offsets[tid];
        int end = d_row_offsets[tid + 1];
        
        for (int edge = start; edge < end; edge++) {
            int source = d_col_indices[edge];
            sum += d_contributions[source];
        }
        
        // PageRank equation: (1-d)/N + d * (sum of PR(i)/outdegree(i))
        float base_value = (1.0f - damping_factor) / num_vertices;
        float damping_value = damping_factor * (sum + dangling_contribution);
        
        d_new_ranks[tid] = base_value + damping_value;
    }
}

// Compute error/difference between consecutive PageRank iterations
__global__ void computeErrorKernel(
    const float* d_old_ranks,
    const float* d_new_ranks,
    float* d_errors,
    vertex_t num_vertices) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vertices) {
        // L1 norm for convergence check
        d_errors[tid] = fabs(d_new_ranks[tid] - d_old_ranks[tid]);
    }
}

// Count out-degrees of each vertex
__global__ void countOutDegreesKernel(
    const vertex_t* d_row_offsets,
    vertex_t* d_degrees,
    vertex_t num_vertices) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vertices) {
        d_degrees[tid] = d_row_offsets[tid + 1] - d_row_offsets[tid];
    }
}

// Implementation of CSR graph creation from edge list
void convertToCSR(
    const std::vector<std::pair<vertex_t, vertex_t>>& edges,
    const std::vector<weight_t>& weights,
    vertex_t num_vertices,
    CSRGraph& graph) {
    
    // 1. Reset and initialize graph properties
    graph.freeMemory();
    graph.num_vertices = num_vertices;
    graph.num_edges = edges.size();
    graph.has_weights = !weights.empty();
    
    // 2. Count outgoing edges for each vertex
    std::vector<vertex_t> edge_counts(num_vertices, 0);
    for (const auto& edge : edges) {
        if (edge.first >= 0 && edge.first < num_vertices) {
            edge_counts[edge.first]++;
        }
    }
    
    // 3. Create row offsets array
    graph.h_row_offsets.resize(num_vertices + 1);
    graph.h_row_offsets[0] = 0;
    for (vertex_t i = 0; i < num_vertices; i++) {
        graph.h_row_offsets[i + 1] = graph.h_row_offsets[i] + edge_counts[i];
    }
    
    // 4. Create column indices and values arrays
    graph.h_col_indices.resize(graph.num_edges);
    if (graph.has_weights) {
        graph.h_values.resize(graph.num_edges);
    }
    
    // Reset edge counts for use as cursor
    std::fill(edge_counts.begin(), edge_counts.end(), 0);
    
    // Fill column indices and values
    for (size_t i = 0; i < edges.size(); i++) {
        const auto& edge = edges[i];
        if (edge.first >= 0 && edge.first < num_vertices) {
            vertex_t vertex = edge.first;
            size_t offset = graph.h_row_offsets[vertex] + edge_counts[vertex];
            
            graph.h_col_indices[offset] = edge.second;
            if (graph.has_weights) {
                graph.h_values[offset] = weights[i];
            }
            
            edge_counts[vertex]++;
        }
    }
}

// Allocate memory for CSR graph on device
void allocateGraphMemory(CSRGraph& graph) {
    // Allocate row offsets array
    CHECK_CUDA(cudaMalloc(&graph.d_row_offsets, 
                          (graph.num_vertices + 1) * sizeof(vertex_t)));
    
    // Allocate column indices array
    CHECK_CUDA(cudaMalloc(&graph.d_col_indices, 
                          graph.num_edges * sizeof(edge_t)));
    
    // Allocate values array if needed
    if (graph.has_weights) {
        CHECK_CUDA(cudaMalloc(&graph.d_values, 
                              graph.num_edges * sizeof(weight_t)));
    }
}

// Copy graph data from host to device
void copyGraphToDevice(CSRGraph& graph) {
    // Copy row offsets
    CHECK_CUDA(cudaMemcpy(graph.d_row_offsets, graph.h_row_offsets.data(),
                          (graph.num_vertices + 1) * sizeof(vertex_t),
                          cudaMemcpyHostToDevice));
    
    // Copy column indices
    CHECK_CUDA(cudaMemcpy(graph.d_col_indices, graph.h_col_indices.data(),
                          graph.num_edges * sizeof(edge_t),
                          cudaMemcpyHostToDevice));
    
    // Copy values if needed
    if (graph.has_weights && !graph.h_values.empty()) {
        CHECK_CUDA(cudaMemcpy(graph.d_values, graph.h_values.data(),
                              graph.num_edges * sizeof(weight_t),
                              cudaMemcpyHostToDevice));
    }
}

// Create a small synthetic graph for testing PageRank
CSRGraph createTestGraph() {
    // Define a small directed graph (adjacency list representation)
    //   0 -> 1, 2
    //   1 -> 0, 3
    //   2 -> 0, 3
    //   3 -> 0
    // In the resulting PageRank, vertex 0 should have the highest rank
    
    // Define edges
    std::vector<std::pair<vertex_t, vertex_t>> edges = {
        {0, 1}, {0, 2},  // Vertex 0 points to 1 and 2
        {1, 0}, {1, 3},  // Vertex 1 points to 0 and 3
        {2, 0}, {2, 3},  // Vertex 2 points to 0 and 3
        {3, 0}           // Vertex 3 points to 0
    };
    
    // No weights for this test graph
    std::vector<weight_t> weights;
    
    // Create CSR graph
    CSRGraph graph;
    vertex_t num_vertices = 4;
    convertToCSR(edges, weights, num_vertices, graph);
    
    // Allocate and copy to device
    allocateGraphMemory(graph);
    copyGraphToDevice(graph);
    
    return graph;
}

// Create a large random graph for performance testing
CSRGraph createLargeTestGraph() {
    // Generate a large random graph
    const vertex_t num_vertices = 100000;
    const float density = 0.0001f; // This will give approximately 1M edges
    
    std::cout << "Generating random graph with " << num_vertices << " vertices...\n";
    
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<vertex_t> dist(0, num_vertices - 1);
    
    // Generate random edges
    std::vector<std::pair<vertex_t, vertex_t>> edges;
    size_t estimated_edges = size_t(num_vertices) * 
                            (size_t(10) + size_t(num_vertices) * density);
    edges.reserve(estimated_edges);
    
    for (vertex_t v = 0; v < num_vertices; v++) {
        // Ensure each vertex has at least one outgoing edge
        edges.emplace_back(v, dist(gen));
        
        // Add additional random edges based on density
        int num_extra_edges = std::binomial_distribution<int>(
            num_vertices - 1, density)(gen);
            
        for (int i = 0; i < num_extra_edges; i++) {
            vertex_t target = dist(gen);
            if (target != v) { // Avoid self-loops
                edges.emplace_back(v, target);
            }
        }
    }
    
    std::cout << "Generated " << edges.size() << " edges\n";
    
    // Create CSR graph
    CSRGraph graph;
    std::vector<weight_t> weights; // No weights for this test
    convertToCSR(edges, weights, num_vertices, graph);
    
    // Allocate and copy to device
    allocateGraphMemory(graph);
    copyGraphToDevice(graph);
    
    return graph;
}
// PageRank implementation for CSR graphs
void computePageRank(
    const CSRGraph& graph,
    std::vector<float>& ranks,
    const PageRankConfig& config) {
    
    // Initialize variables
    Timer timer;
    timer.start();
    
    vertex_t num_vertices = graph.num_vertices;
    int block_size = config.block_size;
    int num_blocks = (num_vertices + block_size - 1) / block_size;
    
    // Allocate device memory
    float *d_old_ranks, *d_new_ranks, *d_errors, *d_contributions;
    vertex_t *d_degrees;
    
    CHECK_CUDA(cudaMalloc(&d_old_ranks, num_vertices * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_new_ranks, num_vertices * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_errors, num_vertices * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_contributions, num_vertices * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_degrees, num_vertices * sizeof(vertex_t)));
    
    // Count out-degrees for each vertex
    countOutDegreesKernel<<<num_blocks, block_size>>>(
        graph.d_row_offsets, d_degrees, num_vertices);
    CHECK_CUDA(cudaGetLastError());
    
    // Initialize PageRank values
    initializePageRankKernel<<<num_blocks, block_size>>>(
        d_old_ranks, num_vertices);
    CHECK_CUDA(cudaGetLastError());
    
    // PageRank iteration loop
    float max_error = config.tolerance + 1.0f;  // Ensure at least one iteration
    int iteration = 0;
    
    std::cout << "\nStarting PageRank iterations:\n";
    std::cout << "------------------------------------\n";
    std::cout << "Iter | Max Change  | Time (ms)\n";
    std::cout << "------------------------------------\n";
    
    Timer iter_timer;
    
    while (max_error > config.tolerance && iteration < config.max_iterations) {
        iter_timer.start();
        
        // Compute contributions for each vertex
        computeContributionsKernel<<<num_blocks, block_size>>>(
            graph.d_row_offsets, d_degrees, d_old_ranks, d_contributions, num_vertices);
        CHECK_CUDA(cudaGetLastError());
        
        // Calculate dangling node contribution (simplified approach)
        // In a more complete implementation, we'd sum up the ranks of nodes with no outgoing edges
        float dangling_contribution = 0.0f;
        
        // PageRank iteration
        pageRankIterationKernel<<<num_blocks, block_size>>>(
            graph.d_row_offsets, graph.d_col_indices, d_contributions,
            d_new_ranks, config.damping_factor, dangling_contribution, num_vertices);
        CHECK_CUDA(cudaGetLastError());
        
        // Compute error/change
        computeErrorKernel<<<num_blocks, block_size>>>(
            d_old_ranks, d_new_ranks, d_errors, num_vertices);
        CHECK_CUDA(cudaGetLastError());
        
        // Find maximum error (using Thrust for reduction)
        thrust::device_ptr<float> d_errors_ptr(d_errors);
        max_error = thrust::reduce(d_errors_ptr, d_errors_ptr + num_vertices, 
                                  0.0f, thrust::maximum<float>());
        
        // Swap old and new ranks
        std::swap(d_old_ranks, d_new_ranks);
        
        iter_timer.stop();
        
        // Print progress
        std::cout << std::setw(4) << iteration << " | " 
                  << std::scientific << std::setprecision(3) << max_error << " | "
                  << std::fixed << std::setprecision(2) << iter_timer.elapsedMilliseconds() 
                  << std::endl;
        
        iteration++;
    }
    
    std::cout << "------------------------------------\n";
    std::cout << "PageRank converged after " << iteration << " iterations\n";
    
    // Copy final results back to host
    ranks.resize(num_vertices);
    CHECK_CUDA(cudaMemcpy(ranks.data(), d_old_ranks,
                         num_vertices * sizeof(float),
                         cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA(cudaFree(d_old_ranks));
    CHECK_CUDA(cudaFree(d_new_ranks));
    CHECK_CUDA(cudaFree(d_errors));
    CHECK_CUDA(cudaFree(d_contributions));
    CHECK_CUDA(cudaFree(d_degrees));

    timer.stop();
    std::cout << "Total computation time: " << timer.elapsedMilliseconds() << " ms\n";
}

} // namespace graphx

int main() {
    try {
        // Uncomment to run with small test graph
        /*
        std::cout << "Creating small test graph...\n";
        graphx::CSRGraph graph = graphx::createTestGraph();

        std::cout << "Graph created with " << graph.num_vertices << " vertices and "
                  << graph.num_edges << " edges\n";

        // Configure PageRank parameters
        graphx::PageRankConfig config;
        config.damping_factor = 0.85f;
        config.tolerance = 1e-6f;
        config.max_iterations = 100;
        config.block_size = 256;

        // Compute PageRank
        std::vector<float> ranks;
        graphx::computePageRank(graph, ranks, config);

        // Print results
        std::cout << "\nFinal PageRank values:\n";
        std::cout << "--------------------\n";
        std::cout << "Vertex | PageRank\n";
        std::cout << "--------------------\n";
        for (size_t i = 0; i < ranks.size(); i++) {
            std::cout << std::setw(6) << i << " | " 
                      << std::fixed << std::setprecision(6) << ranks[i] << "\n";
        }
        std::cout << "--------------------\n";

        // Verify that PageRank values sum to approximately 1
        float sum = 0.0f;
        for (float rank : ranks) {
            sum += rank;
        }
        std::cout << "\nSum of PageRank values: " << sum 
                  << " (should be approximately 1.0)\n";
        
        */
        
        // Run with large test graph for performance testing
        std::cout << "Creating large test graph...\n";
        Timer total_timer;
        total_timer.start();
        
        Timer graph_gen_timer;
        graph_gen_timer.start();
        graphx::CSRGraph graph = graphx::createLargeTestGraph();
        graph_gen_timer.stop();
        
        std::cout << "Graph creation time: " << graph_gen_timer.elapsedMilliseconds() << " ms\n";
        std::cout << "Graph created with " << graph.num_vertices << " vertices and "
                  << graph.num_edges << " edges\n";

        // Configure PageRank parameters
        graphx::PageRankConfig config;
        config.damping_factor = 0.85f;
        config.tolerance = 1e-6f;
        config.max_iterations = 100;
        config.block_size = 256;
        
        // Compute PageRank
        std::vector<float> ranks;
        graphx::computePageRank(graph, ranks, config);
        
        // Print summary statistics instead of all values
        std::cout << "\nPageRank Statistics:\n";
        std::cout << "-------------------\n";
        
        // Calculate min, max, and average PageRank
        float min_rank = *std::min_element(ranks.begin(), ranks.end());
        float max_rank = *std::max_element(ranks.begin(), ranks.end());
        float avg_rank = std::accumulate(ranks.begin(), ranks.end(), 0.0f) / ranks.size();
        
        std::cout << "Minimum PageRank: " << min_rank << "\n";
        std::cout << "Maximum PageRank: " << max_rank << "\n";
        std::cout << "Average PageRank: " << avg_rank << "\n";
        std::cout << "Sum of PageRank values: " 
                  << std::accumulate(ranks.begin(), ranks.end(), 0.0f) << "\n";
        
        total_timer.stop();
        std::cout << "\nTotal execution time: " << total_timer.elapsedMilliseconds() << " ms\n";
        

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
