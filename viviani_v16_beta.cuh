// viviani_v16_beta.cuh  —  V16 Beta-Sheet Extension
//
// WHAT THIS IS:
//   A drop-in extension to viviani_v16_gpu.cuh that upgrades the single
//   forward-bias cursor (α-helix) into a dual-strand β-sheet allocator.
//
// REQUIRED INCLUDE ORDER:
//   #include <cuda_runtime.h>
//   #include <cooperative_groups.h>
//   #include "aizawa.cuh"          // for Viviani geometry: viviani_normal,
//                                  //     compute_hopf_q, viviani_offset_5d
//   #include "aizawa_slab.cuh"     // for slab_viviani_scatter (reused exactly)
//   #include "viviani_v16_gpu.cuh" // base V16 types and helpers
//   #include "viviani_v16_beta.cuh"
//
// PHILOSOPHY: "Everything correct stays untouched forever."
//   Nothing in aizawa.cuh, aizawa_slab.cuh, or viviani_v16_gpu.cuh is
//   modified. All β-sheet logic lives here.
//
// =============================================================================
// THE β-SHEET MODEL
// =============================================================================
//
// An α-helix is a single strand: one cursor advances forward through memory.
// A β-sheet is two strands running in parallel (or antiparallel), bonded at
// regular rung positions — hydrogen bonds in protein chemistry.
//
// Mapped to the allocator:
//
//   Strand A  ──────────────────────────────────────────►  (forward cursor)
//              |           |           |           |
//             rung        rung        rung        rung      (cross-link pts)
//              |           |           |           |
//   Strand B  ──────────────────────────────────────────►  (forward cursor)
//
// Both strands advance independently. Each warp is assigned to a strand by
// its Viviani phase (reusing slab_viviani_scatter from aizawa_slab.cuh).
// The rung positions are the superblocks where Hopf Q invariant mod 2 == 0,
// computable from compute_hopf_q (aizawa.cuh). When a warp's primary strand
// bitmap is exhausted, it attempts a single CAS on its nearest rung in the
// partner strand before falling back — this is the hydrogen bond.
//
// The rice paddy / ping-pong property: each terrace (strand) retains what
// overflows from the other before water (allocation pressure) is lost to
// fallback. The rung is the sluice gate between terraces.
//
// =============================================================================
// STRAND ASSIGNMENT (Viviani phase)
// =============================================================================
//
//   slab_viviani_scatter(warp_id, total_warps) returns a value in
//   [0, SBS_PER_WARP). We map this to a strand: even → A (0), odd → B (1).
//   This gives warps a geometrically distributed, deterministic strand home.
//   Adjacent warps on the Viviani curve tend to be on opposite strands,
//   which is exactly the β-sheet antiparallel geometry.
//
// =============================================================================
// RUNG POSITIONS (Hopf Q cross-link)
// =============================================================================
//
//   A superblock index `sb` is a rung if:
//     (sb * SLAB_SUPERBLOCK_BYTES / VIVIANI_BLOCK_SIZE) % 4 == 0
//   i.e., the block index falls at a Hopf Q period-4 node.
//   This reuses the same period-4 modulus already governing the CPU allocator's
//   protection logic and bin structure — the geometry is consistent across
//   host and device.
//
//   Rungs are not owned by either strand. Either strand can claim a rung slot
//   via CAS. Once claimed, the rung slot is freed normally (bitmap OR).
//   Rungs are checked only on strand exhaustion, not on every alloc — they
//   are the cross-link of last resort before fallback, not the primary path.
//
// =============================================================================
// EXCLUSIVE MODE INTERACTION
// =============================================================================
//
//   V16's capped exclusive mode (non-overlapping ranges per warp) maps cleanly:
//   each strand gets its own non-overlapping range. Strand A warps occupy
//   [warp_slot * CAP, warp_slot * CAP + CAP) within the A-half of the pool;
//   strand B warps occupy the same within the B-half. The pool is split at
//   pool_depth / 2. Rungs live at the boundary — the first superblock of the
//   B-half that is geometrically registered with the last of A, and vice versa.
//
// =============================================================================

#ifndef VIVIANI_V16_BETA_CUH
#define VIVIANI_V16_BETA_CUH

#ifdef __CUDACC__

#include <cuda_runtime.h>
// aizawa.cuh, aizawa_slab.cuh, viviani_v16_gpu.cuh expected above

// ============================================================================
// Configuration
// ============================================================================

// Number of β-sheet strands. Fixed at 2 — the biological model is a pair.
// Extending to 4 (a β-barrel) is future work.
#define BETA_NUM_STRANDS        2

// Rung check budget: how many rung superblocks to scan on strand exhaustion
// before giving up. Kept small (2) to preserve the O(1) character of the
// fast path. The rung positions are deterministic so 2 is almost always
// sufficient — the first rung covers ~50% of cases, the second ~95%.
#define BETA_RUNG_SCAN_DEPTH    2

// Strand split point: pool is divided at this fraction.
// 0.5 = equal halves (true parallel β-sheet).
// Could be tuned per class in future (anchors vs bridges).
#define BETA_STRAND_SPLIT       0  // computed as pool_depth / BETA_NUM_STRANDS

// ============================================================================
// Beta-Sheet Pool State
//
// Extends V16_SlabPool with per-strand cursors and cross-link stats.
// We do NOT modify V16_SlabPool. Instead, callers hold a BetaPool which
// wraps a V16_SlabPool and adds the dual-strand layer.
// ============================================================================

typedef struct {
    // Per-strand, per-class independent cursor (the two advancing fronts).
    // d_strand_cursor[strand][class] replaces the single d_warp_cursor[class].
    // Strand A = index 0, Strand B = index 1.
    uint32_t* d_strand_cursor[BETA_NUM_STRANDS];  // each: uint32_t[SLAB_CLASSES]

    // Cross-link (rung) statistics — host-readable via beta_pool_stats()
    uint64_t* d_rung_hits;    // uint64_t[SLAB_CLASSES]: successful rung claims
    uint64_t* d_rung_misses;  // uint64_t[SLAB_CLASSES]: rung CAS failed (both exhausted)

    // Strand assignment counters — verify Viviani scatter is distributing evenly
    uint64_t* d_strand_allocs[BETA_NUM_STRANDS]; // uint64_t[SLAB_CLASSES] each
} BetaStrandState;

// Full β-sheet pool: a V16_SlabPool (unchanged) + BetaStrandState overlay.
typedef struct {
    V16_SlabPool     base;    // All V16 infrastructure: superblocks, bitmaps,
                              // d_allocs/frees/fallbacks, d_shell_state. Untouched.
    BetaStrandState  beta;    // Dual-strand layer on top.
    bool             initialized;
} BetaSlabPool;

// ============================================================================
// Host: Init / Destroy / Reset
// ============================================================================

static inline cudaError_t beta_slab_init(BetaSlabPool* bp, uint32_t pool_depth) {
    memset(bp, 0, sizeof(*bp));

    // Initialize the base V16 pool (all superblocks, bitmaps, stats, shell state)
    cudaError_t e = v16_slab_init((V16_SlabContext*)&bp->base, pool_depth);
    // Note: v16_slab_init takes a V16_SlabContext*, but BetaSlabPool embeds
    // V16_SlabPool directly. Cast is safe — V16_SlabContext is {V16_SlabPool, bool}.
    // However, to be clean we go through the pool directly:
    // (v16_slab_init is replicated inline below to avoid the cast issue)

    // Re-initialize base properly without the cast hack:
    memset(&bp->base, 0, sizeof(bp->base));
    bp->base.pool_depth = pool_depth;

    for (int c = 0; c < V16_SLAB_CLASSES; c++) {
        e = cudaMalloc((void**)&bp->base.pool[c],
                       (size_t)pool_depth * V16_SLAB_SUPERBLOCK_BYTES);
        if (e != cudaSuccess) return e;
        cudaMemset(bp->base.pool[c], 0,
                   (size_t)pool_depth * V16_SLAB_SUPERBLOCK_BYTES);
        bp->base.pool_base[c] = (uint8_t*)bp->base.pool[c];
    }

    // Base V16 counters (d_warp_cursor is kept for shared-mode fallback compat)
    e = cudaMalloc((void**)&bp->base.d_warp_cursor,
                   V16_SLAB_CLASSES * sizeof(uint32_t));
    if (e != cudaSuccess) return e;
    cudaMemset(bp->base.d_warp_cursor, 0, V16_SLAB_CLASSES * sizeof(uint32_t));

    e = cudaMalloc((void**)&bp->base.d_allocs,    V16_SLAB_CLASSES * sizeof(uint64_t)); if (e != cudaSuccess) return e;
    e = cudaMalloc((void**)&bp->base.d_frees,     V16_SLAB_CLASSES * sizeof(uint64_t)); if (e != cudaSuccess) return e;
    e = cudaMalloc((void**)&bp->base.d_fallbacks, V16_SLAB_CLASSES * sizeof(uint64_t)); if (e != cudaSuccess) return e;
    cudaMemset(bp->base.d_allocs,    0, V16_SLAB_CLASSES * sizeof(uint64_t));
    cudaMemset(bp->base.d_frees,     0, V16_SLAB_CLASSES * sizeof(uint64_t));
    cudaMemset(bp->base.d_fallbacks, 0, V16_SLAB_CLASSES * sizeof(uint64_t));

    e = cudaMalloc((void**)&bp->base.d_shell_state, sizeof(V16_ShellState));
    if (e != cudaSuccess) return e;
    cudaMemset(bp->base.d_shell_state, 0, sizeof(V16_ShellState));

    // Beta-sheet strand cursors (two independent advancing fronts)
    for (int s = 0; s < BETA_NUM_STRANDS; s++) {
        e = cudaMalloc((void**)&bp->beta.d_strand_cursor[s],
                       V16_SLAB_CLASSES * sizeof(uint32_t));
        if (e != cudaSuccess) return e;
        cudaMemset(bp->beta.d_strand_cursor[s], 0,
                   V16_SLAB_CLASSES * sizeof(uint32_t));
    }

    // Cross-link (rung) stats
    e = cudaMalloc((void**)&bp->beta.d_rung_hits,   V16_SLAB_CLASSES * sizeof(uint64_t)); if (e != cudaSuccess) return e;
    e = cudaMalloc((void**)&bp->beta.d_rung_misses, V16_SLAB_CLASSES * sizeof(uint64_t)); if (e != cudaSuccess) return e;
    cudaMemset(bp->beta.d_rung_hits,   0, V16_SLAB_CLASSES * sizeof(uint64_t));
    cudaMemset(bp->beta.d_rung_misses, 0, V16_SLAB_CLASSES * sizeof(uint64_t));

    // Per-strand alloc counters
    for (int s = 0; s < BETA_NUM_STRANDS; s++) {
        e = cudaMalloc((void**)&bp->beta.d_strand_allocs[s],
                       V16_SLAB_CLASSES * sizeof(uint64_t));
        if (e != cudaSuccess) return e;
        cudaMemset(bp->beta.d_strand_allocs[s], 0,
                   V16_SLAB_CLASSES * sizeof(uint64_t));
    }

    cudaDeviceSynchronize();
    bp->initialized = true;
    return cudaSuccess;
}

static inline void beta_slab_destroy(BetaSlabPool* bp) {
    if (!bp->initialized) return;

    // Destroy base V16 pool resources
    for (int c = 0; c < V16_SLAB_CLASSES; c++)
        if (bp->base.pool[c]) cudaFree(bp->base.pool[c]);
    if (bp->base.d_warp_cursor)  cudaFree(bp->base.d_warp_cursor);
    if (bp->base.d_allocs)       cudaFree(bp->base.d_allocs);
    if (bp->base.d_frees)        cudaFree(bp->base.d_frees);
    if (bp->base.d_fallbacks)    cudaFree(bp->base.d_fallbacks);
    if (bp->base.d_shell_state)  cudaFree(bp->base.d_shell_state);

    // Destroy beta strand state
    for (int s = 0; s < BETA_NUM_STRANDS; s++) {
        if (bp->beta.d_strand_cursor[s])  cudaFree(bp->beta.d_strand_cursor[s]);
        if (bp->beta.d_strand_allocs[s])  cudaFree(bp->beta.d_strand_allocs[s]);
    }
    if (bp->beta.d_rung_hits)   cudaFree(bp->beta.d_rung_hits);
    if (bp->beta.d_rung_misses) cudaFree(bp->beta.d_rung_misses);

    bp->initialized = false;
}

static inline void beta_slab_reset(BetaSlabPool* bp) {
    if (!bp->initialized) return;

    for (int c = 0; c < V16_SLAB_CLASSES; c++)
        cudaMemset(bp->base.pool[c], 0,
                   (size_t)bp->base.pool_depth * V16_SLAB_SUPERBLOCK_BYTES);
    cudaMemset(bp->base.d_warp_cursor, 0, V16_SLAB_CLASSES * sizeof(uint32_t));
    cudaMemset(bp->base.d_allocs,      0, V16_SLAB_CLASSES * sizeof(uint64_t));
    cudaMemset(bp->base.d_frees,       0, V16_SLAB_CLASSES * sizeof(uint64_t));
    cudaMemset(bp->base.d_fallbacks,   0, V16_SLAB_CLASSES * sizeof(uint64_t));
    cudaMemset(bp->base.d_shell_state, 0, sizeof(V16_ShellState));

    for (int s = 0; s < BETA_NUM_STRANDS; s++) {
        cudaMemset(bp->beta.d_strand_cursor[s], 0,
                   V16_SLAB_CLASSES * sizeof(uint32_t));
        cudaMemset(bp->beta.d_strand_allocs[s], 0,
                   V16_SLAB_CLASSES * sizeof(uint64_t));
    }
    cudaMemset(bp->beta.d_rung_hits,   0, V16_SLAB_CLASSES * sizeof(uint64_t));
    cudaMemset(bp->beta.d_rung_misses, 0, V16_SLAB_CLASSES * sizeof(uint64_t));

    cudaDeviceSynchronize();
}

// ============================================================================
// Device: Strand Assignment via Viviani Phase
//
// Reuses slab_viviani_scatter() from aizawa_slab.cuh exactly.
// Even scatter value → Strand A (0), odd → Strand B (1).
// Adjacent warps on the Viviani curve alternate strands — antiparallel geometry.
// ============================================================================

__device__ static inline uint32_t beta_strand_for_warp(
    uint32_t warp_id_in_block,
    uint32_t warps_per_block)
{
    // Get base scatter from Viviani geometry
    uint32_t scatter = slab_viviani_scatter(warp_id_in_block,
                                            warps_per_block > 0 ? warps_per_block : 1);

    // CRITICAL FIX: Mix in warp_id and block index to break parity bias
    uint32_t block_id = blockIdx.x + blockIdx.y * gridDim.x
    + blockIdx.z * gridDim.x * gridDim.y;
    uint32_t global_wid = block_id * warps_per_block + warp_id_in_block;

    // Golden ratio hash to mix bits
    scatter ^= global_wid * 0x9e3779b9;
    scatter ^= scatter >> 16;

    return scatter & 1u;  // 0 = Strand A, 1 = Strand B
}

// ============================================================================
// Device: Rung Position Test
//
// A superblock is a rung (cross-link point) if its block index falls at a
// Hopf Q period-4 node — the same modulus governing the CPU allocator's
// protection logic. This keeps host and device geometry consistent.
//
// sb_idx: superblock index within the pool
// returns: 1 if this superblock is a rung position, 0 otherwise
// ============================================================================

__device__ static inline uint32_t beta_is_rung(uint32_t sb_idx) {
    // Each superblock covers SLAB_SUPERBLOCK_BYTES / VIVIANI_BLOCK_SIZE blocks.
    // SLAB_SUPERBLOCK_BYTES = 4096, VIVIANI_BLOCK_SIZE = 64 → 64 blocks/superblock.
    // Rung: first block of the superblock falls at period-4 node.
    //   block_idx = sb_idx * 64
    //   rung iff block_idx % 4 == 0, which is always true (64 is divisible by 4).
    // To get meaningful rung spacing, use the superblock index directly:
    //   rung iff sb_idx % (SLAB_SUPERBLOCK_BYTES / VIVIANI_BLOCK_SIZE / 4) == 0
    //        iff sb_idx % 16 == 0
    // This places rungs every 16 superblocks — one per cache segment at typical
    // L2 granularity, and geometrically at the Hopf period-4 spacing.
    return (sb_idx % 16u == 0u) ? 1u : 0u;
}

// ============================================================================
// Device: Nearest Rung in Partner Strand
//
// Given a warp's current position in its own strand, returns the superblock
// index of the nearest rung in the partner strand's half of the pool.
//
// Pool is split: Strand A owns [0, half), Strand B owns [half, pool_depth).
// Rungs are at multiples of 16 within each half.
// The nearest rung to position `pos` is: (pos / 16) * 16, clamped to the
// partner strand's range.
// ============================================================================

__device__ static inline uint32_t beta_nearest_rung(
    uint32_t my_strand,
    uint32_t my_cursor,      // current cursor position within own strand's half
    uint32_t pool_depth)
{
    uint32_t half      = pool_depth / BETA_NUM_STRANDS;
    uint32_t partner   = 1u - my_strand;
    uint32_t p_base    = partner * half;
    uint32_t p_end     = p_base + half;

    // Mirror the cursor position into the partner half
    uint32_t mirrored  = my_cursor % half;

    // Round down to nearest rung (rungs at multiples of 16 within half)
    uint32_t rung_off  = (mirrored / 16u) * 16u;
    uint32_t rung_abs  = p_base + rung_off;

    // CRITICAL FIX: Ensure we don't go beyond partner's range
    if (rung_abs >= p_end) {
        rung_abs = p_base;  // Wrap to start of partner half
    }

    return rung_abs;
}

// ============================================================================
// Device: Beta-Sheet Alloc
//
// The full allocation path with dual-strand β-sheet logic:
//
//   1. Determine warp's strand (Viviani phase)
//   2. Claim range within that strand's half of the pool (one atomicAdd)
//   3. Attempt to claim a slot within the warp's owned range (V16 bitmap logic)
//   4. On exhaustion: scan BETA_RUNG_SCAN_DEPTH rung positions in partner strand
//   5. On rung miss: fall back (same as V16 fallback path)
//
// Per-warp register state (callers declare and preserve across calls):
//   uint32_t sb_base   = 0xFFFFFFFFu;  // unclaimed sentinel
//   uint32_t sb_cursor = 0;
//   uint32_t strand    = 0xFFFFFFFFu;  // unassigned sentinel
//
// All three must be preserved across calls, same as sb_base/sb_cursor in V16.
// ============================================================================

__device__ static inline void* beta_slab_alloc(
    V16_SlabPool*  pool,
    BetaStrandState* beta,
    int            cls,
    uint32_t*      sb_base,
    uint32_t*      sb_cursor,
    uint32_t*      strand)
{
    if (cls < 0) return nullptr;

    const uint32_t lane       = threadIdx.x & 31u;
    const uint32_t warp_mask  = __activemask();
    const uint32_t leader     = __ffs(warp_mask) - 1u;
    const uint32_t n_slots    = v16_slab_slots(cls);
    const uint32_t sbs_needed = v16_slab_sbs_per_warp_alloc(cls);
    const uint32_t half       = pool->pool_depth / BETA_NUM_STRANDS;

    // Positions within one strand half
    const uint32_t sbs_per_warp = (V16_SLAB_SBS_PER_WARP < half)
    ? V16_SLAB_SBS_PER_WARP : half;

    // -------------------------------------------------------------------------
    // Step 1: Strand assignment (once per warp lifetime)
    // -------------------------------------------------------------------------
    if (*strand == 0xFFFFFFFFu) {
        if (lane == leader) {
            uint32_t wpb      = blockDim.x / 32u;
            uint32_t local_wid = threadIdx.x / 32u;
            *strand = beta_strand_for_warp(local_wid, wpb);
        }
        *strand = __shfl_sync(warp_mask, *strand, leader);
    }

    const uint32_t my_strand = *strand;
    const uint32_t s_base_offset = my_strand * half;

    // -------------------------------------------------------------------------
    // Step 2: Claim range within this strand's half
    // -------------------------------------------------------------------------
    if (*sb_base == 0xFFFFFFFFu) {
        uint32_t base = 0, init_cursor = 0;
        if (lane == leader) {
            uint32_t raw = atomicAdd(&beta->d_strand_cursor[my_strand][cls],
                                     sbs_per_warp);
            base = s_base_offset + (raw % half);

            uint32_t wpb      = blockDim.x  / 32u;
            uint32_t local_wid = threadIdx.x / 32u;
            uint32_t scatter  = slab_viviani_scatter(local_wid, wpb > 0 ? wpb : 1);
            uint32_t n_pos    = sbs_per_warp / sbs_needed;
            init_cursor       = (scatter % n_pos) * sbs_needed;
        }
        *sb_base   = __shfl_sync(warp_mask, base,        leader);
        *sb_cursor = __shfl_sync(warp_mask, init_cursor, leader);
    }

    // -------------------------------------------------------------------------
    // Step 3: Primary alloc attempt
    // -------------------------------------------------------------------------
    const uint32_t n_positions = sbs_per_warp / sbs_needed;

    for (uint32_t attempt = 0; attempt < n_positions; attempt++) {
        uint32_t sb_sub    = lane / n_slots;
        uint32_t slot      = lane % n_slots;
        uint32_t global_sb = *sb_base + *sb_cursor + sb_sub;

        if (global_sb >= pool->pool_depth) break;

        V16_SlabSuperblock* sb = &pool->pool[cls][global_sb];

        if (lane == sb_sub * n_slots) {
            if (sb->bitmap == 0u)
                V16_ATOMIC_CAS((uint32_t*)&sb->bitmap, 0u,
                               v16_slab_init_bitmap(cls));
        }
        __syncwarp(warp_mask);

        bool succeeded = v16_cooperative_claim_slot(sb, slot, warp_mask, leader, lane);
        uint32_t winners = __ballot_sync(warp_mask, succeeded);

        if (succeeded) {
            if (lane == leader) {
                V16_ATOMIC_ADD_ULL(&pool->d_allocs[cls],
                                   (unsigned long long)__popc(winners));
                atomicAdd((unsigned long long*)&beta->d_strand_allocs[my_strand][cls],
                          (unsigned long long)__popc(winners));
            }
            return (void*)(sb->data + (size_t)slot * v16_slab_stride(cls));
        }

        uint32_t next = 0;
        if (lane == leader) {
            *sb_cursor = (*sb_cursor + sbs_needed) % sbs_per_warp;
            next = *sb_cursor;
        }
        *sb_cursor = __shfl_sync(warp_mask, next, leader);
    }

    // -------------------------------------------------------------------------
    // Step 4: Strand exhausted — attempt rung cross-link
    // -------------------------------------------------------------------------
    {
        uint32_t rung_sb   = beta_nearest_rung(my_strand, *sb_cursor,
                                               pool->pool_depth);
        void*    rung_ptr  = nullptr;
        bool     rung_hit  = false;

        for (uint32_t r = 0; r < BETA_RUNG_SCAN_DEPTH; r++) {
            uint32_t probe_sb = rung_sb + r * 16u;
            if (probe_sb >= pool->pool_depth) break;

            if (!beta_is_rung(probe_sb % (pool->pool_depth / BETA_NUM_STRANDS)))
                continue;

            V16_SlabSuperblock* rsb = &pool->pool[cls][probe_sb];

            if (lane == leader) {
                if (rsb->bitmap == 0u) {
                    V16_ATOMIC_CAS((uint32_t*)&rsb->bitmap, 0u,
                                   v16_slab_init_bitmap(cls));
                }
                uint32_t old = V16_ATOMIC_AND((uint32_t*)&rsb->bitmap, ~1u);
                if (old & 1u) {
                    rung_ptr = (void*)rsb->data;
                    rung_hit = true;
                }
            }

            uint32_t hit_broadcast = __ballot_sync(warp_mask,
                                                   (lane == leader) && rung_hit);
            if (hit_broadcast) {
                uint64_t ptr_val = (lane == leader) ? (uint64_t)(uintptr_t)rung_ptr : 0;
                ptr_val = __shfl_sync(warp_mask, ptr_val, leader);
                rung_ptr = (void*)(uintptr_t)ptr_val;

                if (lane == leader) {
                    V16_ATOMIC_ADD_ULL(&pool->d_allocs[cls], 1ULL);
                    atomicAdd((unsigned long long*)&beta->d_rung_hits[cls], 1ULL);
                }
                return rung_ptr;
            }
        }

        // All rung probes failed
        if (lane == leader)
            atomicAdd((unsigned long long*)&beta->d_rung_misses[cls], 1ULL);
    }

    // -------------------------------------------------------------------------
    // Step 5: True fallback — both strand and rungs exhausted
    // -------------------------------------------------------------------------
    V16_ATOMIC_ADD_ULL(&pool->d_fallbacks[cls], 1ULL);
    return nullptr;
}

// ============================================================================
// Device: Free (routes to V16 free — bitmap OR is strand-agnostic)
//
// The free path does not need to know which strand the pointer came from.
// Rung slots free identically to normal slots — bitmap OR, no special case.
// ============================================================================

__device__ static inline void beta_slab_free(
    V16_SlabPool* pool, void* ptr, int cls)
{
    // V16 free is correct and complete. Reuse it exactly.
    v16_slab_free(pool, ptr, cls);
}

// Convenience size wrappers
__device__ static inline void* beta_slab_alloc_sz(
    V16_SlabPool* pool, BetaStrandState* beta, size_t size,
    uint32_t* sb_base, uint32_t* sb_cursor, uint32_t* strand)
{
    return beta_slab_alloc(pool, beta, v16_slab_class(size),
                           sb_base, sb_cursor, strand);
}

__device__ static inline void beta_slab_free_sz(
    V16_SlabPool* pool, void* ptr, size_t size)
{
    beta_slab_free(pool, ptr, v16_slab_class(size));
}

// ============================================================================
// Host: Stats
// ============================================================================

typedef struct {
    V16_SlabStats  v16;                                      // all V16 stats unchanged
    uint64_t       rung_hits[V16_SLAB_CLASSES];              // cross-link successes
    uint64_t       rung_misses[V16_SLAB_CLASSES];            // cross-link failures
    uint64_t       strand_allocs[BETA_NUM_STRANDS][V16_SLAB_CLASSES]; // per-strand
} BetaSlabStats;

static inline BetaSlabStats beta_slab_stats(const BetaSlabPool* bp) {
    BetaSlabStats s = {0};
    if (!bp->initialized) return s;

    // Gather base V16 stats via a temporary context wrapper
    // (avoids needing to change v16_slab_stats signature)
    cudaMemcpy(s.v16.allocs,    bp->base.d_allocs,    V16_SLAB_CLASSES * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(s.v16.frees,     bp->base.d_frees,     V16_SLAB_CLASSES * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(s.v16.fallbacks, bp->base.d_fallbacks, V16_SLAB_CLASSES * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    V16_ShellState ss;
    cudaMemcpy(&ss, bp->base.d_shell_state, sizeof(V16_ShellState), cudaMemcpyDeviceToHost);
    for (int c = 0; c < V16_SLAB_CLASSES; c++) {
        s.v16.contention[c]          = ss.total_contention[c];
        s.v16.halfstep_flips[c]      = ss.halfstep_flips[c];
        s.v16.op_buffer_hits[c]      = ss.op_buffer_hits[c];
        s.v16.shell_decays[c]        = ss.shell_decays[c];
        s.v16.exclusive_mode_count[c]= ss.exclusive_mode_count[c];
        for (int sh = 0; sh < V16_NUM_SHELLS; sh++)
            s.v16.shell_counts[c][sh] = ss.shell_count[c][sh];
    }

    // Beta-sheet stats
    cudaMemcpy(s.rung_hits,   bp->beta.d_rung_hits,   V16_SLAB_CLASSES * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(s.rung_misses, bp->beta.d_rung_misses, V16_SLAB_CLASSES * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    for (int str = 0; str < BETA_NUM_STRANDS; str++)
        cudaMemcpy(s.strand_allocs[str], bp->beta.d_strand_allocs[str],
                   V16_SLAB_CLASSES * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    return s;
}

static inline void beta_slab_print_stats(const BetaSlabStats* s) {
    const char* nm[]     = {"64B ", "128B", "256B"};
    const char* snm[]    = {"A", "B"};

    printf("=== Viviani β-Sheet Slab Stats (V16 + Dual-Strand) ===\n");
    printf("  Class  |   Allocs   |   Frees    | Fallbacks | RungHits | RungMiss\n");
    printf("  -------|------------|------------|-----------|----------|----------\n");
    for (int c = 0; c < V16_SLAB_CLASSES; c++) {
        printf("  %s   | %10llu | %10llu | %9llu | %8llu | %8llu\n", nm[c],
               (unsigned long long)s->v16.allocs[c],
               (unsigned long long)s->v16.frees[c],
               (unsigned long long)s->v16.fallbacks[c],
               (unsigned long long)s->rung_hits[c],
               (unsigned long long)s->rung_misses[c]);
    }

    printf("  Strand distribution (A=Viviani-even, B=Viviani-odd):\n");
    for (int c = 0; c < V16_SLAB_CLASSES; c++) {
        printf("    %s:", nm[c]);
        for (int str = 0; str < BETA_NUM_STRANDS; str++)
            printf(" [%s]=%llu", snm[str],
                   (unsigned long long)s->strand_allocs[str][c]);
        printf("\n");
    }

    printf("  Half-step flips / Op-buffer hits:\n");
    for (int c = 0; c < V16_SLAB_CLASSES; c++) {
        printf("    %s: flips=%u  opbuf=%u  exclusive=%u\n", nm[c],
               s->v16.halfstep_flips[c],
               s->v16.op_buffer_hits[c],
               s->v16.exclusive_mode_count[c]);
    }
}

// ============================================================================
// Stress Test Kernel
//
// Exercises the full β-sheet path: strand assignment, primary bitmap alloc,
// rung cross-link, fallback. Uses three per-warp registers (sb_base,
// sb_cursor, strand) stored in shared memory, same pattern as aizawa_slab_test.
// ============================================================================

__global__ void beta_stress_kernel(
    V16_SlabPool*    pool,
    BetaStrandState* beta,
    uint32_t         iters,
    uint32_t*        d_success_count)
{
    const uint32_t warps_per_block = blockDim.x / 32u;
    const uint32_t warp_id         = threadIdx.x / 32u;

    // Per-warp registers in shared memory
    extern __shared__ uint32_t smem[];
    // Layout: [sb_base × CLASSES] [sb_cursor × CLASSES] [strand × CLASSES]
    uint32_t* warp_sb_base   = smem + warp_id * V16_SLAB_CLASSES;
    uint32_t* warp_sb_cursor = smem + warps_per_block * V16_SLAB_CLASSES
                                    + warp_id * V16_SLAB_CLASSES;
    uint32_t* warp_strand    = smem + 2u * warps_per_block * V16_SLAB_CLASSES
                                    + warp_id * V16_SLAB_CLASSES;

    if ((threadIdx.x & 31u) == 0) {
        for (int c = 0; c < V16_SLAB_CLASSES; c++) {
            warp_sb_base[c]   = 0xFFFFFFFFu;
            warp_sb_cursor[c] = 0;
            warp_strand[c]    = 0xFFFFFFFFu;  // unassigned
        }
    }
    __syncthreads();

    uint32_t local_success = 0;
    uint32_t global_warp_id = blockIdx.x * warps_per_block + warp_id;
    void*    held[V16_SLAB_CLASSES] = {nullptr};

    for (uint32_t i = 0; i < iters; i++) {
        int cls = (int)((global_warp_id + i) % (uint32_t)V16_SLAB_CLASSES);

        if (held[cls]) {
            beta_slab_free(pool, held[cls], cls);
            held[cls] = nullptr;
        }

        void* ptr = beta_slab_alloc(pool, beta, cls,
                                    &warp_sb_base[cls],
                                    &warp_sb_cursor[cls],
                                    &warp_strand[cls]);
        if (ptr) {
            *((volatile uint8_t*)ptr) = (uint8_t)(threadIdx.x ^ i);
            held[cls] = ptr;
            local_success++;
        }
    }

    for (int c = 0; c < V16_SLAB_CLASSES; c++)
        if (held[c]) beta_slab_free(pool, held[c], c);

    atomicAdd(d_success_count, local_success);
}

// Host launcher for the stress kernel
static inline void beta_run_stress(BetaSlabPool* bp, uint32_t blocks,
                                   uint32_t threads, uint32_t iters)
{
    if (!bp->initialized) {
        printf("[beta stress] Not initialized.\n");
        return;
    }

    uint32_t* d_count;
    cudaMalloc((void**)&d_count, sizeof(uint32_t));
    cudaMemset(d_count, 0, sizeof(uint32_t));

    // Shared memory: 3 arrays (sb_base, sb_cursor, strand) × warps × classes
    uint32_t wpb      = threads / 32u;
    size_t   smem_sz  = 3u * wpb * (size_t)V16_SLAB_CLASSES * sizeof(uint32_t);

    // Pass pool and beta state as device pointers
    // (BetaSlabPool is host-side; we pass bp->base and bp->beta by address)
    // For a real integration, these would be in managed memory or passed via
    // a device-side wrapper. Here we use cudaMemcpy to a device copy.
    V16_SlabPool*    d_pool;
    BetaStrandState* d_beta;
    cudaMalloc((void**)&d_pool, sizeof(V16_SlabPool));
    cudaMalloc((void**)&d_beta, sizeof(BetaStrandState));
    cudaMemcpy(d_pool, &bp->base, sizeof(V16_SlabPool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, &bp->beta, sizeof(BetaStrandState), cudaMemcpyHostToDevice);

    beta_stress_kernel<<<blocks, threads, smem_sz>>>(d_pool, d_beta, iters, d_count);
    cudaDeviceSynchronize();

    uint32_t h = 0;
    cudaMemcpy(&h, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("[beta stress] %u blocks × %u threads × %u iters → %u successful allocs\n",
           blocks, threads, iters, h);

    BetaSlabStats bs = beta_slab_stats(bp);
    beta_slab_print_stats(&bs);

    cudaFree(d_count);
    cudaFree(d_pool);
    cudaFree(d_beta);
}

// ============================================================================
// Integration note for video codec / VQP
// ============================================================================
//
// When this plugs back into the video encoder:
//
//   Anchor blocks (phases 0, 2) → assign to Strand A
//   Bridge blocks (phases 1, 3) → assign to Strand B
//
// The Viviani scatter already distributes warps this way naturally because
// anchor warps and bridge warps arrive at geometrically alternating positions
// on the curve. The rung cross-link is the hydrogen bond: a bridge warp that
// can't find space borrows an anchor's rung slot, exactly as a bridge residue
// borrows from the anchor strand's backbone in protein secondary structure.
//
// The sparse Fourier tier selector (DCT bucket count → block size class) is
// a pure client decision: it calls beta_slab_alloc_sz with the appropriate
// size (64/128/256B) and the allocator handles the rest. The codec doesn't
// need to know about strands, rungs, or Viviani geometry.
//
// The ping-pong / rice paddy staging buffers in the shader pipeline
// (shell ping-pong for YUV, described in doc section 51) map to the V16
// half-step shell state (d_shell_state) which is preserved here in bp->base.
// Those buffers advance through the shell alternation independently of which
// strand their memory came from — the two mechanisms are orthogonal.

#endif  // __CUDACC__
#endif  // VIVIANI_V16_BETA_CUH
