// compare_strategies.cu
// Quick comparison: V16 single-strand vs Beta dual-strand
//
// Compile:
//   nvcc -O2 -std=c++17 -arch=sm_75 -o compare compare_strategies.cu -lm
//   ./compare

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "aizawa.cuh"
#include "aizawa_slab.cuh"
#include "viviani_v16_gpu.cuh"
#include "viviani_v16_beta.cuh"

// ============================================================================
// Configuration
// ============================================================================
#define ITERS_PER_WARP    1000
#define WARMUP_ITERS      100
#define MEASURE_ITERS     10

typedef struct {
    const char* name;
    float alloc_rate;        // success rate
    float imbalance;         // strand imbalance (0-1)
    uint64_t rung_hits;      // cross-link successes
    uint64_t rung_misses;    // cross-link failures
    uint64_t fallbacks;      // true fallbacks
    float avg_cycles;        // average cycles per alloc
} StrategyResult;

// ============================================================================
// Timing kernel for V16
// ============================================================================
__global__ void bench_v16_kernel(V16_SlabPool* pool, int iters,
                                 unsigned long long* d_cycles,
                                 unsigned long long* d_success,
                                 unsigned long long* d_fail)
{
    const uint32_t lane = threadIdx.x & 31u;
    uint32_t sb_base = 0xFFFFFFFFu;
    uint32_t sb_cursor = 0;

    unsigned long long ok = 0, fail = 0;
    unsigned long long start = 0, end = 0;

    if (lane == 0) start = clock64();

    for (int i = 0; i < iters; i++) {
        void* p = v16_slab_alloc(pool, 0, &sb_base, &sb_cursor);
        if (p) {
            ok++;
            if (lane == 0) ((uint32_t*)p)[0] = threadIdx.x;
            __syncwarp(__activemask());
            v16_slab_free(pool, p, 0);
        } else {
            fail++;
        }
    }

    if (lane == 0) {
        end = clock64();
        atomicAdd(d_cycles, end - start);
        atomicAdd(d_success, ok);
        atomicAdd(d_fail, fail);
    }
}

// ============================================================================
// Timing kernel for Beta
// ============================================================================
__global__ void bench_beta_kernel(V16_SlabPool* pool, BetaStrandState* beta,
                                  int iters,
                                  unsigned long long* d_cycles,
                                  unsigned long long* d_success,
                                  unsigned long long* d_fail)
{
    const uint32_t lane = threadIdx.x & 31u;

    uint32_t sb_base = 0xFFFFFFFFu;
    uint32_t sb_cursor = 0;
    uint32_t strand = 0xFFFFFFFFu;

    unsigned long long ok = 0, fail = 0;
    unsigned long long start = 0, end = 0;

    if (lane == 0) start = clock64();

    for (int i = 0; i < iters; i++) {
        void* p = beta_slab_alloc(pool, beta, 0, &sb_base, &sb_cursor, &strand);
        if (p) {
            ok++;
            if (lane == 0) ((uint32_t*)p)[0] = threadIdx.x;
            __syncwarp(__activemask());
            beta_slab_free(pool, p, 0);
        } else {
            fail++;
        }
    }

    if (lane == 0) {
        end = clock64();
        atomicAdd(d_cycles, end - start);
        atomicAdd(d_success, ok);
        atomicAdd(d_fail, fail);
    }
}

// ============================================================================
// Run comparison at different pressure levels
// ============================================================================
void run_comparison(uint32_t pool_depth, uint32_t blocks, uint32_t threads) {
    printf("\n=== Comparison: pool_depth=%u, %u blocks × %u threads ===\n",
           pool_depth, blocks, threads);

    // Initialize both allocators
    V16_SlabContext v16_ctx;
    CUDA_CHECK(v16_slab_init(&v16_ctx, pool_depth));

    BetaSlabPool beta_pool;
    CUDA_CHECK(beta_slab_init(&beta_pool, pool_depth));

    // Device buffers for results
    unsigned long long *d_cycles, *d_success, *d_fail;
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_success, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_fail, sizeof(unsigned long long)));

    // Warmup runs
    for (int w = 0; w < WARMUP_ITERS; w++) {
        CUDA_CHECK(cudaMemset(d_cycles, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_success, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_fail, 0, sizeof(unsigned long long)));

        bench_v16_kernel<<<blocks, threads>>>(&v16_ctx.pool, ITERS_PER_WARP,
                                              d_cycles, d_success, d_fail);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemset(d_cycles, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_success, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_fail, 0, sizeof(unsigned long long)));

        bench_beta_kernel<<<blocks, threads>>>(&beta_pool.base, &beta_pool.beta,
                                               ITERS_PER_WARP, d_cycles, d_success, d_fail);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Measure V16
    StrategyResult v16 = {0};
    v16.name = "V16 single-strand";

    CUDA_CHECK(cudaMemset(d_cycles, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_success, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_fail, 0, sizeof(unsigned long long)));

    for (int m = 0; m < MEASURE_ITERS; m++) {
        bench_v16_kernel<<<blocks, threads>>>(&v16_ctx.pool, ITERS_PER_WARP,
                                              d_cycles, d_success, d_fail);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long cycles, success, fail;
    CUDA_CHECK(cudaMemcpy(&cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&success, d_success, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&fail, d_fail, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    uint64_t total_ops = success + fail;
    v16.alloc_rate = (total_ops > 0) ? (float)success / total_ops : 0;
    v16.avg_cycles = (success > 0) ? (float)cycles / success : 0;

    V16_SlabStats v16_stats = v16_slab_stats(&v16_ctx);
    v16.fallbacks = v16_stats.fallbacks[0];

    // Measure Beta
    StrategyResult beta = {0};
    beta.name = "Beta dual-strand";

    CUDA_CHECK(cudaMemset(d_cycles, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_success, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_fail, 0, sizeof(unsigned long long)));

    for (int m = 0; m < MEASURE_ITERS; m++) {
        bench_beta_kernel<<<blocks, threads>>>(&beta_pool.base, &beta_pool.beta,
                                               ITERS_PER_WARP, d_cycles, d_success, d_fail);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&success, d_success, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&fail, d_fail, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    total_ops = success + fail;
    beta.alloc_rate = (total_ops > 0) ? (float)success / total_ops : 0;
    beta.avg_cycles = (success > 0) ? (float)cycles / success : 0;

    BetaSlabStats beta_stats = beta_slab_stats(&beta_pool);
    beta.fallbacks = beta_stats.v16.fallbacks[0];
    beta.rung_hits = beta_stats.rung_hits[0];
    beta.rung_misses = beta_stats.rung_misses[0];

    // Calculate imbalance
    uint64_t A = beta_stats.strand_allocs[0][0];
    uint64_t B = beta_stats.strand_allocs[1][0];
    float total = (float)(A + B);
    beta.imbalance = (total > 0) ? fabsf((float)A - (float)B) / total : 0;

    // Print results
    printf("\n%-20s | Success Rate | Avg Cycles | Fallbacks | Rung Hits | Rung Miss | Imbalance\n", "");
    printf("---------------------|--------------|------------|-----------|-----------|-----------|-----------\n");
    printf("%-20s | %12.1f%% | %10.0f | %9llu | %9s | %9s | %9s\n",
           v16.name, v16.alloc_rate * 100, v16.avg_cycles,
           (unsigned long long)v16.fallbacks, "N/A", "N/A", "N/A");
    printf("%-20s | %12.1f%% | %10.0f | %9llu | %9llu | %9llu | %8.1f%%\n",
           beta.name, beta.alloc_rate * 100, beta.avg_cycles,
           (unsigned long long)beta.fallbacks,
           (unsigned long long)beta.rung_hits,
           (unsigned long long)beta.rung_misses,
           beta.imbalance * 100);

    // Cleanup
    cudaFree(d_cycles);
    cudaFree(d_success);
    cudaFree(d_fail);
    v16_slab_destroy(&v16_ctx);
    beta_slab_destroy(&beta_pool);
}

// ============================================================================
// Main - run at different pressure levels
// ============================================================================
int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== V16 vs Beta Strategy Comparison ===\n");
    printf("Device: %s\n\n", prop.name);

    // Test 1: Light pressure - plenty of pool
    run_comparison(256, 2, 128);   // 256 superblocks, 2 blocks × 128 threads = 8 warps

    // Test 2: Medium pressure - moderate contention
    run_comparison(64, 4, 256);    // 64 superblocks, 4 blocks × 256 threads = 32 warps

    // Test 3: High pressure - pool exhausted, rungs should trigger
    run_comparison(32, 8, 256);    // 32 superblocks, 8 blocks × 256 threads = 64 warps

    // Test 4: Extreme pressure - forced fallbacks
    run_comparison(16, 16, 256);   // 16 superblocks, 16 blocks × 256 threads = 128 warps

    return 0;
}
