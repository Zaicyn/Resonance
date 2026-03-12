// compare_strategies_resonance.cu - FIXED VERSION
// Tests V16 vs Beta at different SBS_PER_WARP values, focusing on 27/16 resonance
//
// Compile:
//   # Baseline (18)
//   nvcc -O2 -std=c++17 -arch=sm_75 -o compare_baseline compare_strategies_resonance.cu -lm
//
//   # 24 config
//   nvcc -O2 -std=c++17 -arch=sm_75 -DV16_SLAB_SBS_PER_WARP=24 -o compare_24 compare_strategies_resonance.cu -lm
//
//   # Resonance (27)
//   nvcc -O2 -std=c++17 -arch=sm_75 -DUSE_RESONANCE -o compare_resonance compare_strategies_resonance.cu -lm
//
// Run:
//   ./compare_baseline | tee baseline_18.txt
//   ./compare_24 | tee baseline_24.txt
//   ./compare_resonance | tee resonance_27.txt

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

// ============================================================================
// CUDA error checking macro
// ============================================================================
#define CUDA_CHECK(call) \
do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at line %d\n", \
        cudaGetErrorString(_e), __LINE__); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Resonance Configuration - MUST come before includes
// ============================================================================
// Perfect resonance ratio 27/16 = 1.6875
// This governs the relationship between:
//   - Microtubule cycle (13 protofilaments × 2 + 1 = 27)
//   - Base-4 period (4² = 16)
//   - Interference window (32 = 2⁵)

#ifdef USE_RESONANCE
// Resonance-tuned values - override defaults
#undef V16_SLAB_SBS_PER_WARP
#undef V16_HALFSTEP_INTERVAL
#undef V16_EXCLUSIVE_CAP

#define V16_SLAB_SBS_PER_WARP    27  // One full microtubule cycle
#define V16_HALFSTEP_INTERVAL     9  // 27/3 = 9 (third-harmonic)
#define V16_EXCLUSIVE_CAP         9  // Match halfstep for symmetry
#endif

// Now include the headers
#include "aizawa.cuh"
#include "aizawa_slab.cuh"
#include "viviani_v16_gpu.cuh"
#include "viviani_v16_beta.cuh"

// ============================================================================
// Test parameters
// ============================================================================
#define ITERS_PER_WARP    500
#define WARMUP_ITERS      50
#define MEASURE_ITERS     5

// Pool depths for different configurations
const uint32_t BASELINE_DEPTHS[] = {
    18,   // 1 chunk
    36,   // 2 chunks
    72,   // 4 chunks
    144,  // 8 chunks
    288   // 16 chunks
};

const uint32_t RESONANCE_DEPTHS[] = {
    27,   // 1 cycle
    54,   // 2 cycles
    108,  // 4 cycles
    216,  // 8 cycles
    432   // 16 cycles
};

typedef struct {
    const char* name;
    uint32_t sbs_per_warp;
    float alloc_rate;
    float imbalance;
    uint64_t rung_hits;
    uint64_t rung_misses;
    uint64_t fallbacks;
    float avg_cycles;
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
// Run single test configuration
// ============================================================================
void run_config(const char* config_name, uint32_t sbs_per_warp,
                const uint32_t* depths, int num_depths,
                uint32_t blocks, uint32_t threads) {

    printf("\n=== %s (SBS_PER_WARP=%d) - %u blocks × %u threads ===\n",
           config_name, sbs_per_warp, blocks, threads);
    printf("Depth | Success V16 | Success Beta | Imbalance | Rung Hits | Fallbacks | Cycles V16 | Cycles Beta\n");
    printf("------|-------------|--------------|-----------|-----------|-----------|------------|------------\n");

    for (int d = 0; d < num_depths; d++) {
        uint32_t pool_depth = depths[d];

        // Initialize allocators
        V16_SlabContext v16_ctx;
        CUDA_CHECK(v16_slab_init(&v16_ctx, pool_depth));

        BetaSlabPool beta_pool;
        CUDA_CHECK(beta_slab_init(&beta_pool, pool_depth));

        // Device buffers
        unsigned long long *d_cycles, *d_success, *d_fail;
        CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_success, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_fail, sizeof(unsigned long long)));

        // Warmup
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

        float v16_rate = (success + fail > 0) ? (float)success / (success + fail) * 100 : 0;
        float v16_cycles_avg = (success > 0) ? (float)cycles / success : 0;

        V16_SlabStats v16_stats = v16_slab_stats(&v16_ctx);

        // Measure Beta
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

        float beta_rate = (success + fail > 0) ? (float)success / (success + fail) * 100 : 0;
        float beta_cycles_avg = (success > 0) ? (float)cycles / success : 0;

        BetaSlabStats beta_stats = beta_slab_stats(&beta_pool);

        // Calculate imbalance
        uint64_t A = beta_stats.strand_allocs[0][0];
        uint64_t B = beta_stats.strand_allocs[1][0];
        float total = (float)(A + B);
        float imbalance = (total > 0) ? fabsf((float)A - (float)B) / total * 100 : 0;

        printf("%4d   | %11.1f%% | %12.1f%% | %9.1f%% | %9llu | %9llu | %10.0f | %10.0f\n",
               pool_depth,
               v16_rate,
               beta_rate,
               imbalance,
               (unsigned long long)beta_stats.rung_hits[0],
               (unsigned long long)beta_stats.v16.fallbacks[0],
               v16_cycles_avg,
               beta_cycles_avg);

        // Cleanup
        cudaFree(d_cycles);
        cudaFree(d_success);
        cudaFree(d_fail);
        v16_slab_destroy(&v16_ctx);
        beta_slab_destroy(&beta_pool);
    }
                }

                // ============================================================================
                // Main
                // ============================================================================
                int main() {
                    cudaDeviceProp prop;
                    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

                    printf("=== V16 vs Beta: Resonance Comparison ===\n");
                    printf("Device: %s\n", prop.name);
                    printf("Perfect resonance ratio: 27/16 = 1.6875\n\n");

                    #ifdef USE_RESONANCE
                    printf("RESONANCE MODE: SBS_PER_WARP=27, HALFSTEP=9\n");
                    uint32_t sbs = 27;
                    const uint32_t* depths = RESONANCE_DEPTHS;
                    int num_depths = sizeof(RESONANCE_DEPTHS) / sizeof(RESONANCE_DEPTHS[0]);
                    #else
                    #if V16_SLAB_SBS_PER_WARP == 24
                    printf("BASELINE MODE: SBS_PER_WARP=24, HALFSTEP=16\n");
                    uint32_t sbs = 24;
                    #else
                    printf("BASELINE MODE: SBS_PER_WARP=18, HALFSTEP=16\n");
                    uint32_t sbs = 18;
                    #endif
                    const uint32_t* depths = BASELINE_DEPTHS;
                    int num_depths = sizeof(BASELINE_DEPTHS) / sizeof(BASELINE_DEPTHS[0]);
                    #endif

                    // Test at different pressure levels
                    uint32_t block_configs[] = {2, 4, 8, 16};
                    uint32_t threads = 256;

                    for (int b = 0; b < 4; b++) {
                        run_config(sbs == 27 ? "Resonance" : (sbs == 24 ? "Baseline 24" : "Baseline 18"),
                                   sbs, depths, num_depths, block_configs[b], threads);
                    }

                    return 0;
                }
