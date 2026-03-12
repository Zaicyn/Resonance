// aizawa_slab.cuh  —  Viviani Geometry-Native Slab Allocator  (v8)
// v7: warp-aggregate stats atomics (optimization C)
// v8: SLAB_ATOMIC_* portability macros for AMD/HIP — zero behavior change on CUDA.
//     All warp-aggregate __ballot_sync/__popc logic from v7 preserved exactly.
//
// ARCHITECTURE — mirrors the CPU forward-bias bump allocator exactly:
//
//   CPU: fork_position advances per alloc. unit_index -> viviani_offset_5d
//        -> arena address. Freed slots recirculate via per-bin freelist cache.
//
//   GPU: Each WARP permanently owns a range of SBS_PER_WARP superblocks,
//        claimed once via atomicAdd (= GPU fork_position). The warp cursor
//        wraps modulo SBS_PER_WARP, so freed bitmap bits are naturally
//        reclaimed on the next rotation — no extra bookkeeping.
//
//   LANE MODEL (the 3D XOR gate):
//     sb_sub = lane / n_slots   → which superblock within the warp group
//     slot   = lane % n_slots   → slot within that superblock
//     All 32 lanes get unique (sb_sub, slot) pairs. No scan. No wasted lanes.
//     This is lossless for all three classes:
//       64B : n_slots=32, sb_sub always 0 — 1 sb per warp alloc
//       128B: n_slots=31, sb_sub ∈ {0,1} — 2 sbs per warp alloc
//       256B: n_slots=15, sb_sub ∈ {0,1,2} — 3 sbs per warp alloc
//
//   VIVIANI SCATTER: warp_id_in_block -> theta -> |nz|*Hopf_Q -> phase offset
//     in [0, SBS_PER_WARP). Keeps adjacent warps from hammering the same
//     superblock on refill, same role as 5D recirculation.
//
//   POOL SIZING: pool_depth >= max_concurrent_warps * SBS_PER_WARP
//     (sbs_per_warp_alloc is automatically handled by cursor arithmetic)
//
// REQUIRED include order:
//   #include <cuda_runtime.h>
//   #include <cooperative_groups.h>
//   #include "aizawa.cuh"
//   #include "aizawa_slab.cuh"
//
// Philosophy: "Everything correct stays untouched forever."

#ifndef VIVIANI_SLAB_CUH
#define VIVIANI_SLAB_CUH

#ifdef __CUDACC__

// aizawa.cuh no longer defines _Atomic as a macro (fixed for CUDA 13+ CCCL
// compatibility), so no push/pop guard is needed here.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
// Belt-and-suspenders: undef _Atomic here too in case GCC 13 stdatomic.h
// was pulled in transitively before this point. CCCL uses _Atomic as a
// template parameter name and the function-like macro form corrupts it.
#ifdef _Atomic
    #undef _Atomic
#endif

// ============================================================================
// Portability: device-side atomic wrappers
//   CUDA:    expands to standard intrinsics — identical to prior behavior.
//   HIP:     hipify-perl maps atomicAdd/CAS/And/Or/Max 1:1; these macros make
//            that translation explicit and auditable in one place.
//   Other:   redefine these macros before including to target any GPU runtime.
// Note: __ballot_sync, __popc, __shfl_sync, __ffs, __activemask, __syncwarp
//       are NOT wrapped here — they require separate HIP headers (hip/hip_runtime.h
//       provides equivalent __hip_* builtins). Flag for future porting pass.
// ============================================================================
#define SLAB_ATOMIC_ADD(ptr, val)      atomicAdd((ptr), (val))
#define SLAB_ATOMIC_AND(ptr, val)      atomicAnd((uint32_t*)(ptr), (val))
#define SLAB_ATOMIC_OR(ptr, val)       atomicOr( (uint32_t*)(ptr), (val))
#define SLAB_ATOMIC_CAS(ptr, cmp, val) atomicCAS((uint32_t*)(ptr), (cmp), (val))
#define SLAB_ATOMIC_MAX_ULL(ptr, val)  atomicMax((unsigned long long*)(ptr), \
                                                  (unsigned long long)(val))
#define SLAB_ATOMIC_ADD_ULL(ptr, val)  atomicAdd((unsigned long long*)(ptr), \
                                                  (unsigned long long)(val))

// ============================================================================
// Configuration
// ============================================================================

#define SLAB_CLASSES            3
#define SLAB_SUPERBLOCK_BYTES   4096

// Superblocks per warp (permanent ownership range).
// Must satisfy: SBS_PER_WARP is a multiple of sbs_per_warp_alloc for each class,
// so that cursor steps of sbs_needed never straddle a range boundary:
//   64B : sbs_needed=1  → divisible by 1 ✓
//   128B: sbs_needed=2  → divisible by 2 ✓
//   256B: sbs_needed=3  → divisible by 3 ✓
// LCM(1,2,3)*3=18: 18/1=18, 18/2=9, 18/3=6 positions per class.
// Capacity (256 blocks x 256 threads = 8 warps/block, hold 4 per thread):
//   64B : 18 pos x 32 = 576/warp x 8 = 4608/block >= 1024 ok
//   128B:  9 pos x 32 = 288/warp x 8 = 2304/block >= 1024 ok
//   256B:  6 pos x 32 = 192/warp x 8 = 1536/block >= 1024 ok
#define SLAB_SBS_PER_WARP       18

// Pool depth per class. Must be >= max_concurrent_warps * SBS_PER_WARP.
// 2048 warps x 18 = 36864; +11% headroom = 40960. ~480 MB.
#define SLAB_POOL_DEPTH         40960

// Viviani constants (must match aizawa.cuh exactly)
#define SLAB_VIVIANI_HOPF_Q     1.97f
#define SLAB_VIVIANI_MODULUS    8

// ============================================================================
// Superblock struct — 4096 bytes exactly
//   [0..3]   volatile uint32_t bitmap  (1=free, 0=in-use)
//   [4..63]  padding
//   [64..]   slot data
// ============================================================================

typedef struct {
    volatile uint32_t bitmap;
    uint32_t          _pad[15];
    uint8_t           data[SLAB_SUPERBLOCK_BYTES - 64];
} SlabSuperblock;

static_assert(sizeof(SlabSuperblock) == SLAB_SUPERBLOCK_BYTES,
              "SlabSuperblock must be 4096 bytes");

// ============================================================================
// Size-class helpers
// ============================================================================

__host__ __device__ static inline int slab_class(size_t size) {
    if (size <= 64)  return 0;
    if (size <= 128) return 1;
    if (size <= 256) return 2;
    return -1;
}

// Slots that fit in data region (4032 bytes), capped at 32 (bitmap width)
__host__ __device__ static inline uint32_t slab_slots(int cls) {
    if (cls == 0) return 32u;   // 32*64  = 2048 <= 4032
    if (cls == 1) return 31u;   // 31*128 = 3968 <= 4032
    return               15u;   // 15*256 = 3840 <= 4032
}

__host__ __device__ static inline uint32_t slab_init_bitmap(int cls) {
    if (cls == 0) return 0xFFFFFFFFu;
    if (cls == 1) return 0x7FFFFFFFu;
    return               0x00007FFFu;
}

__host__ __device__ static inline size_t slab_stride(int cls) {
    return (size_t)64u << (uint32_t)cls;  // 64, 128, 256
}

// How many superblocks a warp needs simultaneously (one per warp alloc call)
// so all 32 lanes each get a unique (sb, slot):
//   sb_sub = lane / n_slots,  slot = lane % n_slots
__host__ __device__ static inline uint32_t slab_sbs_per_warp_alloc(int cls) {
    uint32_t n = slab_slots(cls);
    return (32u + n - 1u) / n;   // ceil(32 / n_slots)
}

// ============================================================================
// Viviani geometry  (matches viviani_normal + viviani_offset_5d in aizawa.cuh)
// ============================================================================

__host__ __device__ static inline void slab_viviani_normal(
        float theta, float* nx, float* ny, float* nz) {
    float s=sinf(theta), c=cosf(theta), s3=sinf(3.f*theta), c3=cosf(3.f*theta);
    float x=s-.5f*s3, y=-c+.5f*c3, z=c*c3;
    float n=sqrtf(x*x+y*y+z*z); if(n<1e-6f)n=1.f;
    *nx=x/n; *ny=y/n; *nz=z/n;
}

// Phase scatter within [0, SBS_PER_WARP): mirrors 5D recirculation.
// Uses both |nz| (Hopf projection) and |nx| (XOR partner) — the 3D gate.
__host__ __device__ static inline uint32_t slab_viviani_scatter(
        uint32_t warp_id, uint32_t total_warps) {
    float theta = 2.f * 3.14159265f * (float)warp_id / (float)total_warps;
    float nx, ny, nz;
    slab_viviani_normal(theta, &nx, &ny, &nz);
    float proj = fabsf(nz) * SLAB_VIVIANI_HOPF_Q;
    uint32_t q  = (uint32_t)((int)(proj * (float)SLAB_VIVIANI_MODULUS)
                              % SLAB_VIVIANI_MODULUS);
    uint32_t xc = (uint32_t)(fabsf(nx) * 4.f) & 3u;  // 0..3
    return (q ^ xc) % (uint32_t)SLAB_SBS_PER_WARP;
}

// ============================================================================
// Pool state
// ============================================================================

typedef struct {
    SlabSuperblock* pool[SLAB_CLASSES];
    uint8_t*        pool_base[SLAB_CLASSES];
    uint32_t        pool_depth;
    uint32_t*       d_warp_cursor;   // uint32_t[SLAB_CLASSES], monotone
    uint64_t*       d_allocs;
    uint64_t*       d_frees;
    uint64_t*       d_fallbacks;
} SlabPool;

typedef struct { SlabPool pool; bool initialized; } VivianiSlabContext;

// ============================================================================
// Host init / destroy
// ============================================================================

static inline cudaError_t viviani_slab_init(VivianiSlabContext* ctx,
                                             uint32_t pool_depth) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->pool.pool_depth = pool_depth;
    for (int c = 0; c < SLAB_CLASSES; c++) {
        cudaError_t e = cudaMalloc((void**)&ctx->pool.pool[c],
                                   (size_t)pool_depth * SLAB_SUPERBLOCK_BYTES);
        if (e != cudaSuccess) return e;
        cudaMemset(ctx->pool.pool[c], 0,
                   (size_t)pool_depth * SLAB_SUPERBLOCK_BYTES);
        ctx->pool.pool_base[c] = (uint8_t*)ctx->pool.pool[c];
    }
    cudaError_t e;
    e=cudaMalloc((void**)&ctx->pool.d_warp_cursor, SLAB_CLASSES*sizeof(uint32_t));
    if(e!=cudaSuccess)return e;
    cudaMemset(ctx->pool.d_warp_cursor, 0, SLAB_CLASSES*sizeof(uint32_t));
    e=cudaMalloc((void**)&ctx->pool.d_allocs,    SLAB_CLASSES*sizeof(uint64_t)); if(e!=cudaSuccess)return e;
    e=cudaMalloc((void**)&ctx->pool.d_frees,     SLAB_CLASSES*sizeof(uint64_t)); if(e!=cudaSuccess)return e;
    e=cudaMalloc((void**)&ctx->pool.d_fallbacks, SLAB_CLASSES*sizeof(uint64_t)); if(e!=cudaSuccess)return e;
    cudaMemset(ctx->pool.d_allocs,    0, SLAB_CLASSES*sizeof(uint64_t));
    cudaMemset(ctx->pool.d_frees,     0, SLAB_CLASSES*sizeof(uint64_t));
    cudaMemset(ctx->pool.d_fallbacks, 0, SLAB_CLASSES*sizeof(uint64_t));
    ctx->initialized = true;
    return cudaSuccess;
}

static inline void viviani_slab_destroy(VivianiSlabContext* ctx) {
    if (!ctx->initialized) return;
    for (int c=0;c<SLAB_CLASSES;c++) if(ctx->pool.pool[c]) cudaFree(ctx->pool.pool[c]);
    if(ctx->pool.d_warp_cursor) cudaFree(ctx->pool.d_warp_cursor);
    if(ctx->pool.d_allocs)      cudaFree(ctx->pool.d_allocs);
    if(ctx->pool.d_frees)       cudaFree(ctx->pool.d_frees);
    if(ctx->pool.d_fallbacks)   cudaFree(ctx->pool.d_fallbacks);
    ctx->initialized = false;
}

// ============================================================================
// Host stats
// ============================================================================

typedef struct { uint64_t allocs[SLAB_CLASSES], frees[SLAB_CLASSES],
                          fallbacks[SLAB_CLASSES]; } SlabStats;

static inline SlabStats viviani_slab_stats(const VivianiSlabContext* ctx) {
    SlabStats s={0};
    if(!ctx->initialized) return s;
    cudaMemcpy(s.allocs,    ctx->pool.d_allocs,    SLAB_CLASSES*sizeof(uint64_t),cudaMemcpyDeviceToHost);
    cudaMemcpy(s.frees,     ctx->pool.d_frees,     SLAB_CLASSES*sizeof(uint64_t),cudaMemcpyDeviceToHost);
    cudaMemcpy(s.fallbacks, ctx->pool.d_fallbacks, SLAB_CLASSES*sizeof(uint64_t),cudaMemcpyDeviceToHost);
    return s;
}

static inline void viviani_slab_print_stats(const SlabStats* s) {
    const char* nm[]={"64B ","128B","256B"};
    printf("=== Viviani Slab Stats ===\n");
    printf("  Class  |   Allocs   |   Frees    | Fallbacks\n");
    printf("  -------|------------|------------|-----------\n");
    for(int c=0;c<SLAB_CLASSES;c++)
        printf("  %s   | %10llu | %10llu | %10llu\n",nm[c],
               (unsigned long long)s->allocs[c],
               (unsigned long long)s->frees[c],
               (unsigned long long)s->fallbacks[c]);
}

// ============================================================================
// Host: Reset
// ============================================================================

static inline void viviani_slab_reset(VivianiSlabContext* ctx) {
    if (!ctx->initialized) return;
    for (int c = 0; c < SLAB_CLASSES; c++)
        cudaMemset(ctx->pool.pool[c], 0,
                   (size_t)ctx->pool.pool_depth * SLAB_SUPERBLOCK_BYTES);
    cudaMemset(ctx->pool.d_warp_cursor, 0, SLAB_CLASSES * sizeof(uint32_t));
    cudaMemset(ctx->pool.d_allocs,      0, SLAB_CLASSES * sizeof(uint64_t));
    cudaMemset(ctx->pool.d_frees,       0, SLAB_CLASSES * sizeof(uint64_t));
    cudaMemset(ctx->pool.d_fallbacks,   0, SLAB_CLASSES * sizeof(uint64_t));
}

// ============================================================================
// Device: Alloc
//
// Per-warp register state (callers must declare and preserve across calls):
//   uint32_t sb_base   = 0xFFFFFFFFu;  // sentinel: unclaimed
//   uint32_t sb_cursor = 0;
//
// Lane model — all 32 lanes covered, zero wasted:
//   n_slots   = slab_slots(cls)              (32, 31, or 15)
//   sbs_needed = slab_sbs_per_warp_alloc(cls) (1,  2,  or 3)
//   For lane L:
//     sb_sub = L / n_slots    <- which of the sbs_needed superblocks
//     slot   = L % n_slots    <- slot within that superblock
//   sb index = sb_base + sb_cursor + sb_sub
//
// Claim: atomicAnd(~(1u<<slot)); check OLD value. If bit was set: we own it.
// If bit was clear (another thread took it): advance cursor by sbs_needed
// and wrap modulo SBS_PER_WARP (recirculation — freed bits come back).
// ============================================================================

__device__ static inline void* viviani_slab_alloc(
    SlabPool*  pool,
    int        cls,
    uint32_t*  sb_base,    // per-warp register (in/out)
    uint32_t*  sb_cursor   // per-warp register (in/out)
) {
    if (cls < 0) return nullptr;

    const uint32_t lane        = threadIdx.x & 31u;
    const uint32_t warp_mask   = __activemask();
    const uint32_t leader      = __ffs(warp_mask) - 1u;
    const uint32_t n_slots     = slab_slots(cls);
    const uint32_t sbs_needed  = slab_sbs_per_warp_alloc(cls);

    // Claim range once (forward bias — one atomicAdd per warp lifetime).
    // sb_base stays at the raw atomicAdd result (always a multiple of
    // SBS_PER_WARP), so the warp's range is exactly [sb_base, sb_base+12).
    // Viviani scatter is applied to the INITIAL CURSOR, snapped to the nearest
    // valid cursor step (a multiple of sbs_needed). This phases adjacent warps
    // apart within their own ranges without any cross-range bleed.
    if (*sb_base == 0xFFFFFFFFu) {
        uint32_t base = 0, init_cursor = 0;
        if (lane == leader) {
            base = SLAB_ATOMIC_ADD(&pool->d_warp_cursor[cls],
                             (uint32_t)SLAB_SBS_PER_WARP);
            uint32_t wid     = threadIdx.x / 32u;
            uint32_t wpb     = blockDim.x  / 32u;
            uint32_t scatter = slab_viviani_scatter(wid, wpb > 0 ? wpb : 1);
            // n_pos valid cursor positions; scatter selects one of them
            uint32_t n_pos   = (uint32_t)SLAB_SBS_PER_WARP / sbs_needed;
            init_cursor      = (scatter % n_pos) * sbs_needed;
        }
        *sb_base   = __shfl_sync(warp_mask, base,        leader);
        *sb_cursor = __shfl_sync(warp_mask, init_cursor, leader);
    }

    // Valid cursor positions: 0, sbs_needed, 2*sbs_needed ...
    // up to SBS_PER_WARP - sbs_needed (inclusive).
    // cursor + sb_sub (max sbs_needed-1) always stays < SBS_PER_WARP.
    // SBS_PER_WARP=18=LCM(1,2,3)*3 guarantees exact divisibility.
    const uint32_t n_positions = (uint32_t)SLAB_SBS_PER_WARP / sbs_needed;

    for (uint32_t attempt = 0; attempt < n_positions; attempt++) {

        uint32_t sb_sub = lane / n_slots;   // 0, 1, or 2
        uint32_t slot   = lane % n_slots;   // 0..n_slots-1

        // cursor is always a multiple of sbs_needed: cursor+sb_sub < SBS_PER_WARP
        uint32_t global_sb = *sb_base + *sb_cursor + sb_sub;
        if (global_sb >= pool->pool_depth) {
            SLAB_ATOMIC_ADD_ULL(&pool->d_fallbacks[cls], 1ULL);
            return nullptr;
        }

        SlabSuperblock* sb = &pool->pool[cls][global_sb];

        // First lane of each sb_sub group stamps its own superblock.
        // 64B: lane 0.  128B: lanes 0, 31.  256B: lanes 0, 15, 30.
        if (lane == sb_sub * n_slots) {
            if (sb->bitmap == 0u)
                SLAB_ATOMIC_CAS((uint32_t*)&sb->bitmap, 0u, slab_init_bitmap(cls));
        }
        __syncwarp(warp_mask);

        // Forward-bias claim: each lane owns exactly one slot in its sb_sub
        uint32_t mask     = 1u << slot;
        uint32_t old_bmap = SLAB_ATOMIC_AND((uint32_t*)&sb->bitmap, ~mask);

        // __ballot_sync MUST be called by ALL lanes unconditionally —
        // it is a warp-wide barrier. Calling it inside a divergent if-block
        // deadlocks lanes that took different branches.
        // So: evaluate success for every lane, then ballot outside any branch.
        bool succeeded = (old_bmap & mask) != 0u;
        uint32_t winners = __ballot_sync(warp_mask, succeeded);
        if (succeeded) {
            // Warp-aggregate: leader posts one atomicAdd for all winners.
            if (lane == leader)
                SLAB_ATOMIC_ADD_ULL(&pool->d_allocs[cls],
                                    (unsigned long long)__popc(winners));
            return (void*)(sb->data + (size_t)slot * slab_stride(cls));
        }

        // All slots at this cursor position taken — step to next, wrap in range
        uint32_t next = 0;
        if (lane == leader) {
            *sb_cursor = (*sb_cursor + sbs_needed) % (uint32_t)SLAB_SBS_PER_WARP;
            next = *sb_cursor;
        }
        *sb_cursor = __shfl_sync(warp_mask, next, leader);
    }

    SLAB_ATOMIC_ADD_ULL(&pool->d_fallbacks[cls], 1ULL);
    return nullptr;
}

// ============================================================================
// Device: Free  (alignment-independent via pool_base)
// ============================================================================

__device__ static inline void viviani_slab_free(
    SlabPool* pool, void* ptr, int cls)
{
    if (cls < 0 || !ptr) return;
    ptrdiff_t off = (uint8_t*)ptr - pool->pool_base[cls];
    if (off < 0) return;
    uint32_t sb_idx = (uint32_t)((size_t)off / SLAB_SUPERBLOCK_BYTES);
    if (sb_idx >= pool->pool_depth) return;
    SlabSuperblock* sb = &pool->pool[cls][sb_idx];
    ptrdiff_t doff = (uint8_t*)ptr - sb->data;
    if (doff < 0) return;
    uint32_t slot = (uint32_t)((size_t)doff / slab_stride(cls));
    if (slot >= slab_slots(cls)) return;
    SLAB_ATOMIC_OR((uint32_t*)&sb->bitmap, 1u << slot);
    // Warp-aggregate: count all lanes that reach the free, one atomicAdd.
    uint32_t free_mask = __activemask();
    uint32_t free_lane = threadIdx.x & 31u;
    uint32_t free_lead = __ffs(free_mask) - 1u;
    if (free_lane == free_lead)
        SLAB_ATOMIC_ADD_ULL(&pool->d_frees[cls],
                            (unsigned long long)__popc(free_mask));
}

// Convenience size_t wrappers
__device__ static inline void* viviani_slab_alloc_sz(
    SlabPool* pool, size_t size, uint32_t* sb_base, uint32_t* sb_cursor)
{ return viviani_slab_alloc(pool, slab_class(size), sb_base, sb_cursor); }

__device__ static inline void viviani_slab_free_sz(
    SlabPool* pool, void* ptr, size_t size)
{ viviani_slab_free(pool, ptr, slab_class(size)); }

// ============================================================================
// Benchmark kernel
// ============================================================================

__global__ void viviani_slab_bench_kernel(SlabPool* pool, int cls,
                                           uint32_t iters, uint64_t* out_cyc) {
    uint32_t sb_base = 0xFFFFFFFFu, sb_cursor = 0;
    uint64_t t0 = clock64();
    void* last = nullptr;
    for (uint32_t i = 0; i < iters; i++) {
        void* ptr = viviani_slab_alloc(pool, cls, &sb_base, &sb_cursor);
        if (ptr) {
            *((volatile uint8_t*)ptr) = (uint8_t)(threadIdx.x + i);
            if (last) viviani_slab_free(pool, last, cls);
            last = ptr;
        }
    }
    if (last) viviani_slab_free(pool, last, cls);
    uint64_t t1 = clock64();
    if (threadIdx.x==0 && blockIdx.x==0)
        SLAB_ATOMIC_MAX_ULL(out_cyc, (unsigned long long)(t1-t0));
}

static inline void viviani_slab_run_bench(VivianiSlabContext* ctx, uint32_t iters) {
    if (!ctx->initialized) { printf("[slab bench] Not initialized.\n"); return; }
    uint64_t* d; cudaMalloc((void**)&d, sizeof(uint64_t));
    const char* nm[]={"64B ","128B","256B"};
    printf("=== Viviani Slab Benchmark (%u iters/thread) ===\n",iters);
    printf("  Class | Cycles/iter | ~GOps/s (est 1.5GHz)\n");
    printf("  ------|-------------|--------------------\n");
    for (int c=0;c<SLAB_CLASSES;c++) {
        cudaMemset(d,0,sizeof(uint64_t));
        viviani_slab_bench_kernel<<<64,128>>>(&ctx->pool,c,iters,d);
        cudaDeviceSynchronize();
        uint64_t h=0; cudaMemcpy(&h,d,sizeof(uint64_t),cudaMemcpyDeviceToHost);
        float cpi=(float)h/iters;
        float gops=(float)((uint64_t)64*128*iters)/((float)h/1.5f)*1e-9f;
        printf("  %s  | %11.1f | %.2f\n",nm[c],cpi,gops);
    }
    cudaFree(d);
    SlabStats _ss = viviani_slab_stats(ctx); viviani_slab_print_stats(&_ss);
}

#endif // __CUDACC__
#endif // VIVIANI_SLAB_CUH
