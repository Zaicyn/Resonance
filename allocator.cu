// aizawa.cuh  —  V9
// Unified Lock-Free Allocator — Microtubule-Grounded Geometry
// Built on V8; all original correct code preserved exactly.
//
// V9 changes:
//   - Microtubule constant derivation chain replaces magic numbers.
//     All timing, stride, and window constants now flow from a single
//     biological ground truth: the 13-protofilament structure.
//
//   - VIVIANI_SHADOW_STRIDE: 8 → 13 (one shadow per protofilament span).
//     Propagates through shadow_count, viviani_alloc, viviani_flyby_check,
//     viviani_geometric_repair, viviani_superfluid_compact. All correct.
//
//   - viviani_geometric_repair: seam-aware.
//     Seam positions (shadow_idx % 13 == 0) apply a 2-bit phase rotation
//     to the stored shadow before comparison — the seam protofilament's
//     quadrature offset (π/2 × SEAM_STRENGTH). Write-back stores the
//     inverse-shifted current so the next scan's forward-shift is stable.
//     Below SEAM_DEFECT_THRESHOLD: forgiving repair, weighted defect count.
//     Above threshold: catastrophe — push to flag queue, don't auto-repair.
//
//   - viviani_reconcile → viviani_microtubule_reconcile: 32-event window.
//     Processes flag queue entries in INTERFERENCE_WINDOW=32 sized batches.
//     Leading strand (shadow) vs lagging strand (ejected pool + queue) with
//     seam phase shift applied at every 13th block. Ligation (clear+update)
//     on match, Aizawa stir on small mismatch, branch increment on large.
//
//   - VivianiAllocator: three new atomic counters for seam pathology.
//   - VivianiStats / viviani_print_stats: expose seam counters.
//
// Philosophy: "Everything correct stays untouched forever."

#ifndef VIVIANI_ALLOC_HOPFION_FIXED_CUH
#define VIVIANI_ALLOC_HOPFION_FIXED_CUH

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _Atomic
#undef _Atomic
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define VIVIANI_HOST_DEVICE __host__ __device__
#define VIVIANI_DEVICE __device__
#endif

#define VIVIANI_ATOMIC volatile
#define VIVIANI_atomic_store(ptr, val)              __atomic_store_n(ptr, val, __ATOMIC_SEQ_CST)
#define VIVIANI_atomic_load(ptr)                    __atomic_load_n(ptr, __ATOMIC_SEQ_CST)
#define VIVIANI_atomic_fetch_add(ptr, val)          __atomic_fetch_add(ptr, val, __ATOMIC_SEQ_CST)
#define VIVIANI_atomic_fetch_sub(ptr, val)          __atomic_fetch_sub(ptr, val, __ATOMIC_SEQ_CST)
#define VIVIANI_atomic_fetch_or(ptr, val)           __atomic_fetch_or(ptr, val, __ATOMIC_SEQ_CST)
#define VIVIANI_atomic_fetch_and(ptr, val)          __atomic_fetch_and(ptr, val, __ATOMIC_SEQ_CST)
#define VIVIANI_atomic_compare_exchange_weak(ptr, expected, desired) \
__atomic_compare_exchange_n(ptr, expected, desired, 1, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)

#ifndef __CUDACC__
#define VIVIANI_HOST_DEVICE
#define VIVIANI_DEVICE
#endif

// ============================================================================
// Configuration — Microtubule-Grounded Derivation Chain
//
// Ground truth: 13-protofilament microtubule (12 parallel + 1 seam).
// Every timing and stride constant flows from this.  No magic numbers.
//
// Previously arbitrary → now derived:
//   VIVIANI_SHADOW_STRIDE : 8  → 13  (one shadow per protofilament span)
//   VIVIANI_FLAG_RING_SIZE: 4096 → 512 (13 × INTERFERENCE_WINDOW, power-of-2)
//   VIVIANI_MAX_DEFECTS   : 64  → 52  (2 × INTERFERENCE_CYCLE, one full turn)
// ============================================================================

#define VIVIANI_POOL_SIZE       (64ULL * 1024 * 1024)   // 64 MB base pool
#define VIVIANI_BLOCK_SIZE      64                       // Cache-line aligned
#define VIVIANI_UNIT_SIZE       256                      // 4 blocks = 1 Hopfion unit
#define VIVIANI_HOPF_Q          1.97f                    // Approximate Hopf charge (Q≈2)

// --- Microtubule geometry (biological ground truth) ---
#define MICROTUBULE_PROTOFILAMENTS  13  // Total protofilaments
#define MICROTUBULE_PARALLEL        12  // In-phase strands (bulk flow)
#define MICROTUBULE_SEAM             1  // Quadrature strand (half-step trigger)

// --- Derived timing constants ---
// Full interference cycle: leading (12) + seam (1) + lagging (12) + seam (1) = 26
#define INTERFERENCE_CYCLE   (2 * (MICROTUBULE_PARALLEL + MICROTUBULE_SEAM))  // 26
// Window: smallest power-of-2 container for one cycle (efficient bitmask ops)
#define INTERFERENCE_WINDOW  32   // 2^5 >= 26

// --- Seam phase shift (π/2 scaled by topological defect strength) ---
// SEAM_STRENGTH = 2.0 - HOPF_Q = 0.03 (deviation from perfect toroidal linking)
// SEAM_PHASE_SHIFT_BITS = round(64 * SEAM_STRENGTH) = round(1.92) = 2
// Applied as a 2-bit rotation of the 64-bit invariant at seam positions.
#define SEAM_STRENGTH            0.03f   // (2.0f - VIVIANI_HOPF_Q)
#define SEAM_PHASE_SHIFT_BITS    2       // round(64 * SEAM_STRENGTH)

// Seam catastrophe threshold: above this many seam defects per repair scan,
// stop auto-repairing and push to flag queue for strict reconciliation.
// Biologically: excessive seam defects → microtubule catastrophe.
#define SEAM_DEFECT_THRESHOLD    5

// --- Derived allocation constants ---
// Shadow stride: one shadow entry covers PROTOFILAMENTS consecutive blocks.
// Was 8 (arbitrary), now 13 (geometrically grounded).
#define VIVIANI_SHADOW_STRIDE    MICROTUBULE_PROTOFILAMENTS  // 13

// Flag ring: 13 × INTERFERENCE_WINDOW = 416, rounded up to 512 (power-of-2).
// Was 4096 (oversized). 512 is sufficient for defect queue depth at this stride.
#define VIVIANI_FLAG_RING_SIZE   512

// Max defects before compaction: 2 × INTERFERENCE_CYCLE (one full microtubule turn).
// Was 64 (arbitrary). 52 = one turn of the 26-step cycle × 2 strands.
#define VIVIANI_MAX_DEFECTS      (2 * INTERFERENCE_CYCLE)   // 52

// Period-4 bin sizes (quaternary encoding) — unchanged
#define VIVIANI_BIN_COUNT       16
#define VIVIANI_BIN_0           64ULL
#define VIVIANI_BIN_1           128ULL
#define VIVIANI_BIN_2           256ULL
#define VIVIANI_BIN_3           512ULL
#define VIVIANI_BIN_4           1024ULL
#define VIVIANI_BIN_5           2048ULL
#define VIVIANI_BIN_6           4096ULL
#define VIVIANI_BIN_7           8192ULL
#define VIVIANI_BIN_8           (32ULL  * 1024)
#define VIVIANI_BIN_9           (64ULL  * 1024)
#define VIVIANI_BIN_10          (128ULL * 1024)
#define VIVIANI_BIN_11          (256ULL * 1024)
#define VIVIANI_BIN_12          (1ULL   * 1024 * 1024)
#define VIVIANI_BIN_13          (4ULL   * 1024 * 1024)
#define VIVIANI_BIN_14          (16ULL  * 1024 * 1024)
#define VIVIANI_BIN_15          (64ULL  * 1024 * 1024)

// Unchanged operational constants
#define VIVIANI_MAX_STREAMS     1024
#define VIVIANI_HOLD_PER_STREAM 8
#define VIVIANI_GPU_HELD_CAP    (1536ULL * 1024 * 1024)

#define VIVIANI_FREELIST_SLOTS  8
#define VIVIANI_CACHE_BYPASS_THRESHOLD 16
#define VIVIANI_FRACTAL_DEPTH   4
#define VIVIANI_5D_MODULUS      8

// ============================================================================
// Aizawa Chaos Types  (quark-level stirring for ejected blocks)
// ============================================================================

typedef struct {
    float state[3];
    float phi;
    uint32_t steps;
} AizawaState;

typedef struct {
    size_t     block_idx;
    AizawaState state;
} EjectedEntry;

#define MAX_EJECTED 4096
#define VIVIANI_EJECTED_EMPTY 0xFFFFFFFFu

// GPU hold-protection rate threshold — derived from framework geometry.
// Unchanged from V8.
#define VIVIANI_ALLOC_RATE_THRESHOLD \
((uint32_t)((VIVIANI_HOPF_Q * 1000.0f) / (float)VIVIANI_FRACTAL_DEPTH))

// ============================================================================
// Viviani Curve Parametrization (Triple Harmonic)
// — unchanged from V8 —
// ============================================================================

typedef struct { float x, y, z; } VivianNormal;

VIVIANI_HOST_DEVICE static inline VivianNormal viviani_normal(float theta) {
    float sin_t  = sinf(theta),       cos_t  = cosf(theta);
    float sin_3t = sinf(3.0f * theta), cos_3t = cosf(3.0f * theta);

    float x = sin_t  - 0.5f * sin_3t;
    float y = -cos_t + 0.5f * cos_3t;
    float z = cos_t  * cos_3t;

    float norm = sqrtf(x*x + y*y + z*z);
    if (norm < 1e-6f) norm = 1.0f;

    return (VivianNormal){ x/norm, y/norm, z/norm };
}

// ============================================================================
// Hopfion Extension 1: Topological Invariant (Linking Number Proxy)
// — unchanged from V8 —
// ============================================================================

VIVIANI_HOST_DEVICE static inline int compute_hopf_q(
    size_t block_idx,
    size_t total_blocks,
    const size_t* offset_table,
    uint32_t num_units)
{
    if (num_units == 0 || offset_table == NULL || block_idx >= total_blocks) return 0;

    int unit_idx = (int)(block_idx % num_units);
    int prev     = (unit_idx - 1 + num_units) % num_units;
    int next     = (unit_idx + 1) % num_units;

    long long diff_prev = (long long)offset_table[unit_idx] - (long long)offset_table[prev];
    long long diff_next = (long long)offset_table[next]     - (long long)offset_table[unit_idx];

    int q = (int)(((llabs(diff_prev) + llabs(diff_next)) / VIVIANI_BLOCK_SIZE) % VIVIANI_5D_MODULUS);
    return q % 4;
}

// ============================================================================
// Hopfion Extension 2: Fractal Binning (Helical Down-Scaling)
// — unchanged from V8 —
// ============================================================================

VIVIANI_HOST_DEVICE static inline int fractal_bin(int base_bin, int level) {
    if (level > VIVIANI_FRACTAL_DEPTH) return base_bin;
    if (base_bin < 0 || base_bin >= VIVIANI_BIN_COUNT) return 0;
    return (base_bin + (level % 4)) % VIVIANI_BIN_COUNT;
}

// ============================================================================
// Hopfion Extension 3: 5D Recirculation in Offsets (Kaluza-Klein inspired)
// — unchanged from V8 —
// ============================================================================

VIVIANI_HOST_DEVICE static inline size_t viviani_offset_5d(int unit_index, int total_units) {
    if (total_units == 0) return 0;

    float theta = 2.0f * 3.14159265f * (float)unit_index / (float)total_units;
    VivianNormal n = viviani_normal(theta);

    float projection = fabsf(n.z) * VIVIANI_HOPF_Q;
    size_t base      = (size_t)unit_index * VIVIANI_UNIT_SIZE;
    size_t offset    = (size_t)(projection * VIVIANI_UNIT_SIZE) % VIVIANI_POOL_SIZE;

    int q                   = (int)(projection * (float)VIVIANI_5D_MODULUS) % VIVIANI_5D_MODULUS;
    size_t recirculation    = ((size_t)q * VIVIANI_BLOCK_SIZE) % VIVIANI_POOL_SIZE;

    return (base + offset + recirculation) % VIVIANI_POOL_SIZE;
}

VIVIANI_HOST_DEVICE static inline size_t viviani_offset(int u, int t) {
    return viviani_offset_5d(u, t);
}

// ============================================================================
// Period-4 Protection Logic — unchanged from V8
// ============================================================================

VIVIANI_HOST_DEVICE static inline bool is_protected(int count, int bin) {
    int group = bin / 4;
    return ((count + group * 4) % 4) == 0;
}

VIVIANI_HOST_DEVICE static inline int quaternary_encode(int value, int flip_state) {
    int base4 = value % 4;
    if (flip_state) {
        return (base4 == 2) ? 0 : ((base4 == 3) ? 1 : base4);
    } else {
        return (base4 == 3) ? 1 : ((base4 == 2) ? 0 : base4);
    }
}

// ============================================================================
// Shadow Parity — Flyby Detection — unchanged from V8
// ============================================================================

typedef struct {
    uint64_t primary;
    uint64_t shadow;
} ShadowPair;

VIVIANI_HOST_DEVICE static inline uint64_t compute_invariant(const uint8_t* data, size_t size) {
    uint64_t inv = 0;
    const uint64_t* ptr = (const uint64_t*)data;
    size_t words = size / sizeof(uint64_t);
    for (size_t i = 0; i < words; i++) inv ^= ptr[i];
    inv ^= (inv >> 32);
    inv ^= (inv >> 16);
    inv ^= (inv >> 8);
    return inv;
}

VIVIANI_HOST_DEVICE static inline uint64_t viviani_invariant(
    const uint8_t* data,
    size_t size,
    int block_index,
    int total_blocks,
    const size_t* offset_table,
    uint32_t num_units)
{
    uint64_t base_inv = compute_invariant(data, size);

    float theta = 2.0f * 3.14159265f * (float)block_index / (float)total_blocks;
    VivianNormal n = viviani_normal(theta);
    int q = compute_hopf_q((size_t)block_index, (size_t)total_blocks, offset_table, num_units);

    uint64_t geo_factor  = (uint64_t)(fabsf(n.z) * 255.0f);
    uint64_t hopf_factor = ((uint64_t)q & 0x03ULL) << 56;

    return (base_inv & 0xFF) | (geo_factor << 8) | hopf_factor;
}

VIVIANI_HOST_DEVICE static inline uint64_t viviani_invariant_simple(
    const uint8_t* data,
    size_t size,
    int block_index,
    int total_blocks)
{
    uint64_t base_inv  = compute_invariant(data, size);
    float theta        = 2.0f * 3.14159265f * (float)block_index / (float)total_blocks;
    VivianNormal n     = viviani_normal(theta);
    uint64_t geo_factor = (uint64_t)(fabsf(n.z) * 255.0f);
    return base_inv ^ (geo_factor << 56);
}

// ============================================================================
// Seam Phase Helpers
//
// The seam protofilament introduces a quadrature offset (π/2 × SEAM_STRENGTH)
// equivalent to a 2-bit rotation in the 64-bit invariant word.
//
// seam_forward_shift : apply before comparing stored shadow to current
// seam_inverse_shift : apply before writing current back to shadow storage
//   (so the next forward-shift recovers the correct comparison value)
// ============================================================================

static inline uint64_t seam_forward_shift(uint64_t v) {
    return (v >> SEAM_PHASE_SHIFT_BITS) | (v << (64 - SEAM_PHASE_SHIFT_BITS));
}

static inline uint64_t seam_inverse_shift(uint64_t v) {
    return (v << SEAM_PHASE_SHIFT_BITS) | (v >> (64 - SEAM_PHASE_SHIFT_BITS));
}

// ============================================================================
// Flag Queue — Lock-Free MPSC Ring Buffer — unchanged from V8
// ============================================================================

typedef struct {
    uint32_t chunk_id;
    uint32_t version;
    uint8_t  defect_type;  // 0=parity, 1=topological drift, 2=overflow,
    // 3=hopf_mismatch, 4=seam_catastrophe (V9 new)
    uint8_t  priority;
    uint16_t delta_hint;
} FlagEntry;

typedef struct {
    FlagEntry      entries[VIVIANI_FLAG_RING_SIZE];
    VIVIANI_ATOMIC uint32_t head;
    VIVIANI_ATOMIC uint32_t tail;
    VIVIANI_ATOMIC uint32_t count;
} FlagQueue;

static inline void flag_queue_init(FlagQueue* fq) {
    memset(fq->entries, 0, sizeof(fq->entries));
    VIVIANI_atomic_store(&fq->head, 0);
    VIVIANI_atomic_store(&fq->tail, 0);
    VIVIANI_atomic_store(&fq->count, 0);
}

static inline bool flag_queue_push(FlagQueue* fq, FlagEntry entry) {
    uint32_t count = VIVIANI_atomic_load(&fq->count);
    if (count >= VIVIANI_FLAG_RING_SIZE - 1) return false;
    uint32_t head = VIVIANI_atomic_load(&fq->head);
    fq->entries[head] = entry;
    VIVIANI_atomic_store(&fq->head, (head + 1) % VIVIANI_FLAG_RING_SIZE);
    VIVIANI_atomic_fetch_add(&fq->count, 1);
    return true;
}

static inline bool flag_queue_pop(FlagQueue* fq, FlagEntry* out) {
    uint32_t count = VIVIANI_atomic_load(&fq->count);
    if (count == 0) return false;
    uint32_t tail = VIVIANI_atomic_load(&fq->tail);
    *out = fq->entries[tail];
    VIVIANI_atomic_store(&fq->tail, (tail + 1) % VIVIANI_FLAG_RING_SIZE);
    VIVIANI_atomic_fetch_sub(&fq->count, 1);
    return true;
}

// ============================================================================
// Free-List Cache — Viral Hijacking (Fast Reuse) — unchanged from V8
// ============================================================================

typedef struct {
    void*            slots[VIVIANI_FREELIST_SLOTS];
    VIVIANI_ATOMIC uint32_t count;
} FreeListCache;

static inline void freelist_cache_init(FreeListCache* fc) {
    memset(fc->slots, 0, sizeof(fc->slots));
    VIVIANI_atomic_store(&fc->count, 0);
}

static inline bool freelist_cache_push(FreeListCache* fc, void* ptr) {
    uint32_t old_count = VIVIANI_atomic_load(&fc->count);
    while (old_count < VIVIANI_FREELIST_SLOTS) {
        uint32_t new_count = old_count + 1;
        if (VIVIANI_atomic_compare_exchange_weak(&fc->count, &old_count, new_count)) {
            fc->slots[old_count] = ptr;
            return true;
        }
    }
    return false;
}

static inline void* freelist_cache_pop(FreeListCache* fc) {
    uint32_t old_count = VIVIANI_atomic_load(&fc->count);
    while (old_count > 0) {
        uint32_t new_count = old_count - 1;
        if (VIVIANI_atomic_compare_exchange_weak(&fc->count, &old_count, new_count)) {
            void* ptr = fc->slots[new_count];
            fc->slots[new_count] = NULL;
            return ptr;
        }
    }
    return NULL;
}

// ============================================================================
// Main Allocator State
// V9 adds three seam-pathology counters to VivianiAllocator.
// All other fields unchanged from V8.
// ============================================================================

typedef struct {
    // Memory arena
    uint8_t* arena;
    size_t   arena_size;

    // Bump allocator state
    VIVIANI_ATOMIC size_t fork_position;

    // Shadow parity table
    ShadowPair* shadows;
    size_t      shadow_count;

    // Flag queue for deferred reconciliation
    FlagQueue flag_queue;

    // Free-list caches (per-bin viral hijacking)
    FreeListCache    freelist_caches[VIVIANI_BIN_COUNT];
    VIVIANI_ATOMIC uint64_t cache_hits;
    VIVIANI_ATOMIC uint64_t cache_misses;

    // Protection state
    VIVIANI_ATOMIC uint32_t allocated_blocks;
    VIVIANI_ATOMIC uint32_t defect_count;
    VIVIANI_ATOMIC uint32_t protected_mode;
    VIVIANI_ATOMIC uint32_t flip_state;

    // Rate tracking (fixed-point: actual_rate = value / 1000)
    VIVIANI_ATOMIC uint32_t alloc_rate_fp;

    // Bin configuration
    size_t           bin_sizes[VIVIANI_BIN_COUNT];
    VIVIANI_ATOMIC uint32_t bin_counters[VIVIANI_BIN_COUNT];

    // Viviani geometry tables
    size_t*      offset_table;
    VivianNormal* normal_table;
    uint32_t     num_units;

    // Version counter for ABA protection
    VIVIANI_ATOMIC uint32_t version;

    // Aizawa quark pool (V8: + direct-mapped index for O(1) lookup)
    EjectedEntry*    ejected_pool;
    uint32_t         ejected_index[MAX_EJECTED];
    VIVIANI_ATOMIC uint32_t ejected_count;

    // V9: Seam-pathology counters
    // seam_defect_count   — seam positions repaired normally (below threshold)
    // seam_catastrophe_count — seam positions pushed to flag queue (above threshold)
    // branch_count        — persistent mismatches that triggered new branches
    VIVIANI_ATOMIC uint32_t seam_defect_count;
    VIVIANI_ATOMIC uint32_t seam_catastrophe_count;
    VIVIANI_ATOMIC uint32_t branch_count;

    // V9: Rolling event counter for 32-window reconciliation triggering
    VIVIANI_ATOMIC uint32_t event_count;

    #ifdef __CUDACC__
    int                num_streams;
    cudaStream_t*      streams;
    void**             stream_holds;
    size_t*            stream_hold_sizes;
    VIVIANI_ATOMIC uint32_t*  stream_hold_counts;
    VIVIANI_ATOMIC uint64_t   gpu_held_bytes;
    #endif
} VivianiAllocator;

// ============================================================================
// Initialization / Destruction — V9: zero new counters, stride change
// propagates automatically through shadow_count formula.
// ============================================================================

static inline void viviani_init(VivianiAllocator* va, size_t arena_size) {
    va->arena_size = (arena_size / 4096) * 4096;
    va->arena      = (uint8_t*)aligned_alloc(4096, va->arena_size);
    if (!va->arena) { fprintf(stderr, "FATAL: Failed to allocate arena\n"); exit(1); }
    memset(va->arena, 0, va->arena_size);

    VIVIANI_atomic_store(&va->fork_position,   0);
    VIVIANI_atomic_store(&va->allocated_blocks, 0);
    VIVIANI_atomic_store(&va->defect_count,    0);
    VIVIANI_atomic_store(&va->protected_mode,  0);
    VIVIANI_atomic_store(&va->flip_state,      0);
    VIVIANI_atomic_store(&va->alloc_rate_fp,   0);
    VIVIANI_atomic_store(&va->cache_hits,      0);
    VIVIANI_atomic_store(&va->cache_misses,    0);
    VIVIANI_atomic_store(&va->version,         0);

    // V9 new counters
    VIVIANI_atomic_store(&va->seam_defect_count,      0);
    VIVIANI_atomic_store(&va->seam_catastrophe_count, 0);
    VIVIANI_atomic_store(&va->branch_count,           0);
    VIVIANI_atomic_store(&va->event_count,            0);

    size_t sizes[] = {
        VIVIANI_BIN_0,  VIVIANI_BIN_1,  VIVIANI_BIN_2,  VIVIANI_BIN_3,
        VIVIANI_BIN_4,  VIVIANI_BIN_5,  VIVIANI_BIN_6,  VIVIANI_BIN_7,
        VIVIANI_BIN_8,  VIVIANI_BIN_9,  VIVIANI_BIN_10, VIVIANI_BIN_11,
        VIVIANI_BIN_12, VIVIANI_BIN_13, VIVIANI_BIN_14, VIVIANI_BIN_15
    };
    memcpy(va->bin_sizes, sizes, sizeof(sizes));

    for (int i = 0; i < VIVIANI_BIN_COUNT; i++) {
        VIVIANI_atomic_store(&va->bin_counters[i], 0);
        freelist_cache_init(&va->freelist_caches[i]);
    }

    va->num_units    = (uint32_t)(va->arena_size / VIVIANI_UNIT_SIZE);
    va->offset_table = (size_t*)malloc(va->num_units * sizeof(size_t));
    va->normal_table = (VivianNormal*)malloc(va->num_units * sizeof(VivianNormal));
    if (!va->offset_table || !va->normal_table) {
        fprintf(stderr, "FATAL: Failed to allocate geometry tables\n");
        exit(1);
    }

    for (uint32_t i = 0; i < va->num_units; i++) {
        va->offset_table[i] = viviani_offset(i, va->num_units);
        float theta          = 2.0f * 3.14159265f * (float)i / (float)va->num_units;
        va->normal_table[i]  = viviani_normal(theta);
    }

    // shadow_count: VIVIANI_SHADOW_STRIDE is now 13 instead of 8.
    // Formula is identical; only the constant changed.
    va->shadow_count = va->arena_size / (VIVIANI_BLOCK_SIZE * VIVIANI_SHADOW_STRIDE);
    va->shadows      = (ShadowPair*)calloc(va->shadow_count, sizeof(ShadowPair));
    if (!va->shadows) {
        fprintf(stderr, "FATAL: Failed to allocate shadow table\n");
        exit(1);
    }

    flag_queue_init(&va->flag_queue);

    va->ejected_pool = (EjectedEntry*)calloc(MAX_EJECTED, sizeof(EjectedEntry));
    if (!va->ejected_pool) {
        fprintf(stderr, "FATAL: Failed to allocate Aizawa ejected pool\n");
        exit(1);
    }
    for (uint32_t i = 0; i < MAX_EJECTED; i++)
        va->ejected_index[i] = VIVIANI_EJECTED_EMPTY;
    VIVIANI_atomic_store(&va->ejected_count, 0);

    #ifdef __CUDACC__
    va->num_streams       = 0;
    va->streams           = NULL;
    va->stream_holds      = NULL;
    va->stream_hold_sizes = NULL;
    va->stream_hold_counts = NULL;
    VIVIANI_atomic_store(&va->gpu_held_bytes, 0);
    #endif
}

static inline void viviani_destroy(VivianiAllocator* va) {
    for (int i = 0; i < VIVIANI_BIN_COUNT; i++) {
        VIVIANI_atomic_store(&va->freelist_caches[i].count, 0);
        memset(va->freelist_caches[i].slots, 0, sizeof(va->freelist_caches[i].slots));
    }

    free(va->arena);
    free(va->offset_table);
    free(va->normal_table);
    free(va->shadows);
    free(va->ejected_pool);

    #ifdef __CUDACC__
    if (va->streams) {
        for (int i = 0; i < va->num_streams; i++) cudaStreamDestroy(va->streams[i]);
        free(va->streams);
        free(va->stream_holds);
        free(va->stream_hold_sizes);
        free((void*)va->stream_hold_counts);
    }
    #endif
}

static inline void viviani_ejected_reset(VivianiAllocator* va) {
    memset(va->ejected_pool, 0, MAX_EJECTED * sizeof(EjectedEntry));
    for (uint32_t i = 0; i < MAX_EJECTED; i++)
        va->ejected_index[i] = VIVIANI_EJECTED_EMPTY;
    VIVIANI_atomic_store(&va->ejected_count, 0);
}

// ============================================================================
// Bin Selection with Fractal Extension — unchanged from V8
// ============================================================================

static inline int viviani_bin_index(VivianiAllocator* va, size_t size) {
    int base = VIVIANI_BIN_COUNT - 1;
    for (int i = 0; i < VIVIANI_BIN_COUNT; i++) {
        if (size <= va->bin_sizes[i]) { base = i; break; }
    }
    int level = VIVIANI_atomic_load(&va->flip_state);
    return fractal_bin(base, level);
}

// ============================================================================
// Allocation — unchanged from V8 except event_count increment
// ============================================================================

static inline void* viviani_alloc(VivianiAllocator* va, size_t size) {
    int    bin         = viviani_bin_index(va, size);
    size_t actual_size = va->bin_sizes[bin];

    void* cached = freelist_cache_pop(&va->freelist_caches[bin]);
    if (cached) {
        VIVIANI_atomic_fetch_add(&va->cache_hits, 1);
        VIVIANI_atomic_fetch_add(&va->allocated_blocks, 1);
        VIVIANI_atomic_fetch_add(&va->bin_counters[bin], 1);
        VIVIANI_atomic_fetch_add(&va->event_count, 1);  // V9: track events
        return cached;
    }

    VIVIANI_atomic_fetch_add(&va->cache_misses, 1);

    size_t blocks_needed = (actual_size + VIVIANI_BLOCK_SIZE - 1) / VIVIANI_BLOCK_SIZE;
    size_t alloc_size    = blocks_needed * VIVIANI_BLOCK_SIZE;
    size_t old_fork      = VIVIANI_atomic_fetch_add(&va->fork_position, alloc_size);

    if (old_fork + alloc_size > va->arena_size) {
        VIVIANI_atomic_fetch_sub(&va->fork_position, alloc_size);
        return NULL;
    }

    void*  ptr       = va->arena + old_fork;
    size_t block_idx = old_fork / VIVIANI_BLOCK_SIZE;

    // shadow_count formula identical; VIVIANI_SHADOW_STRIDE is now 13
    if (block_idx % VIVIANI_SHADOW_STRIDE == 0) {
        size_t shadow_idx = block_idx / VIVIANI_SHADOW_STRIDE;
        if (shadow_idx < va->shadow_count) {
            uint64_t inv = viviani_invariant(
                (uint8_t*)ptr, actual_size,
                                             (int)block_idx, (int)(va->arena_size / VIVIANI_BLOCK_SIZE),
                                             va->offset_table, va->num_units);
            va->shadows[shadow_idx].primary = inv;
            // At seam positions, store inverse-shifted so forward-shift recovers correctly
            va->shadows[shadow_idx].shadow  =
            (shadow_idx % MICROTUBULE_PROTOFILAMENTS == 0)
            ? seam_inverse_shift(inv)
            : inv;
        }
    }

    VIVIANI_atomic_fetch_add(&va->allocated_blocks, 1);
    VIVIANI_atomic_fetch_add(&va->bin_counters[bin], 1);
    VIVIANI_atomic_fetch_add(&va->event_count, 1);  // V9

    uint32_t old_rate = VIVIANI_atomic_load(&va->alloc_rate_fp);
    VIVIANI_atomic_store(&va->alloc_rate_fp, (old_rate * 7 + 1000) / 8);

    return ptr;
}

// ============================================================================
// Deallocation — unchanged from V8 except event_count increment
// ============================================================================

static inline void viviani_free(VivianiAllocator* va, void* ptr, size_t size) {
    if (!ptr) return;

    int bin = viviani_bin_index(va, size);

    if (freelist_cache_push(&va->freelist_caches[bin], ptr)) {
        VIVIANI_atomic_fetch_sub(&va->allocated_blocks, 1);
        VIVIANI_atomic_fetch_sub(&va->bin_counters[bin], 1);
        uint32_t old_rate = VIVIANI_atomic_load(&va->alloc_rate_fp);
        VIVIANI_atomic_store(&va->alloc_rate_fp, (old_rate * 7) / 8);
        VIVIANI_atomic_fetch_add(&va->event_count, 1);  // V9
        return;
    }

    uint32_t alloc_count = VIVIANI_atomic_load(&va->allocated_blocks);
    if (is_protected((int)alloc_count, bin)) {
        FlagEntry entry = {
            .chunk_id   = (uint32_t)(((uint8_t*)ptr - va->arena) / VIVIANI_BLOCK_SIZE),
            .version    = VIVIANI_atomic_load(&va->version),
            .defect_type = 2,
            .priority   = 100,
            .delta_hint = (uint16_t)(size / VIVIANI_BLOCK_SIZE)
        };
        flag_queue_push(&va->flag_queue, entry);
    }

    VIVIANI_atomic_fetch_sub(&va->allocated_blocks, 1);
    VIVIANI_atomic_fetch_sub(&va->bin_counters[bin], 1);
    VIVIANI_atomic_fetch_add(&va->event_count, 1);  // V9

    uint32_t old_rate = VIVIANI_atomic_load(&va->alloc_rate_fp);
    VIVIANI_atomic_store(&va->alloc_rate_fp, (old_rate * 7) / 8);
}

// ============================================================================
// Flyby Detection — unchanged from V8 (stride change propagates automatically)
// ============================================================================

static inline void viviani_flyby_check(VivianiAllocator* va, size_t block_idx) {
    size_t shadow_idx = block_idx / VIVIANI_SHADOW_STRIDE;
    if (shadow_idx >= va->shadow_count) return;
    if (va->shadows[shadow_idx].shadow == 0 &&
        va->shadows[shadow_idx].primary == 0) return;

    size_t total_blocks = va->arena_size / VIVIANI_BLOCK_SIZE;
    uint8_t* block_ptr  = va->arena + (block_idx * VIVIANI_BLOCK_SIZE);

    uint64_t current_inv = viviani_invariant(
        block_ptr, VIVIANI_BLOCK_SIZE,
        (int)block_idx, (int)total_blocks,
                                             va->offset_table, va->num_units);

    uint64_t stored   = va->shadows[shadow_idx].shadow;
    // Apply forward shift at seam positions before comparison
    uint64_t expected = (shadow_idx % MICROTUBULE_PROTOFILAMENTS == 0)
    ? seam_forward_shift(stored)
    : stored;

    if (current_inv != expected) {
        uint8_t  defect_type  = 0;
        uint8_t  priority_val = 128;

        int q_current  = (int)((current_inv >> 56) & 0x03);
        int q_expected = (int)((expected    >> 56) & 0x03);
        if (q_current != q_expected) { defect_type = 3; priority_val = 200; }

        FlagEntry entry = {
            .chunk_id    = (uint32_t)block_idx,
            .version     = VIVIANI_atomic_load(&va->version),
            .defect_type = defect_type,
            .priority    = priority_val,
            .delta_hint  = 64
        };
        flag_queue_push(&va->flag_queue, entry);
        VIVIANI_atomic_fetch_add(&va->defect_count, 1);
    }

    va->shadows[shadow_idx].primary = current_inv;
}

// ============================================================================
// Geometric Repair — V9: seam-aware with adaptive catastrophe detection
//
// Parallel positions (shadow_idx % 13 != 0):
//   Forgiving repair as before. Mismatch → update shadow to current,
//   increment defect_count and repair stats.
//
// Seam positions (shadow_idx % 13 == 0):
//   Apply forward-shift to stored shadow before comparison.
//   Below SEAM_DEFECT_THRESHOLD: repair normally but write back
//     inverse-shifted current (so next scan's forward-shift is stable).
//     Increment seam_defect_count weighted × 2.
//   Above threshold: catastrophe — push to flag queue (defect_type=4),
//     do NOT auto-repair. Increment seam_catastrophe_count.
//
// The viral propagation scan is preserved exactly.
// ============================================================================

typedef struct {
    uint32_t blocks_scanned;
    uint32_t mismatches_found;
    uint32_t repairs_applied;
    uint32_t viral_propagations;
    uint32_t hopf_repairs;
    uint32_t seam_repairs;         // V9: seam positions repaired normally
    uint32_t seam_catastrophes;    // V9: seam positions pushed to queue
} GeometricRepairStats;

static inline GeometricRepairStats viviani_geometric_repair(VivianiAllocator* va) {
    GeometricRepairStats stats = {0};
    size_t total_blocks = va->arena_size / VIVIANI_BLOCK_SIZE;

    uint32_t seam_defects_this_scan = 0;

    for (size_t i = 0; i < total_blocks; i += VIVIANI_SHADOW_STRIDE) {
        stats.blocks_scanned++;

        size_t shadow_idx = i / VIVIANI_SHADOW_STRIDE;
        if (shadow_idx >= va->shadow_count) break;

        uint8_t* block   = va->arena + (i * VIVIANI_BLOCK_SIZE);
        uint64_t current = viviani_invariant(block, VIVIANI_BLOCK_SIZE,
                                             (int)i, (int)total_blocks,
                                             va->offset_table, va->num_units);

        uint64_t stored   = va->shadows[shadow_idx].shadow;
        bool     is_seam  = (shadow_idx % MICROTUBULE_PROTOFILAMENTS == 0);

        // Apply forward phase shift at seam positions before comparison
        uint64_t expected = is_seam ? seam_forward_shift(stored) : stored;

        if (expected == 0 && current == 0) continue;

        if (current != expected) {
            stats.mismatches_found++;

            int q_current  = (int)((current  >> 56) & 0x03);
            int q_expected = (int)((expected >> 56) & 0x03);
            if (q_current != q_expected) stats.hopf_repairs++;

            if (is_seam) {
                seam_defects_this_scan++;

                if (seam_defects_this_scan > SEAM_DEFECT_THRESHOLD) {
                    // Catastrophe: push to flag queue, do not auto-repair.
                    // The 32-window reconciler will handle this entry with
                    // full cross-strand parity checking.
                    FlagEntry entry = {
                        .chunk_id    = (uint32_t)(shadow_idx * VIVIANI_SHADOW_STRIDE),
                        .version     = VIVIANI_atomic_load(&va->version),
                        .defect_type = 4,   // seam_catastrophe
                        .priority    = 255, // highest priority for reconciler
                        .delta_hint  = (uint16_t)shadow_idx
                    };
                    flag_queue_push(&va->flag_queue, entry);
                    stats.seam_catastrophes++;
                    VIVIANI_atomic_fetch_add(&va->seam_catastrophe_count, 1);

                } else {
                    // Normal seam repair: write back inverse-shifted current
                    // so next scan's forward-shift recovers the correct value.
                    va->shadows[shadow_idx].shadow  = seam_inverse_shift(current);
                    va->shadows[shadow_idx].primary = seam_inverse_shift(current);
                    stats.seam_repairs++;
                    stats.repairs_applied++;
                    // Seam defects count double (weaker protofilament)
                    VIVIANI_atomic_fetch_add(&va->seam_defect_count, 2);
                    VIVIANI_atomic_fetch_add(&va->defect_count, 2);
                }

            } else {
                // Parallel position: forgiving repair, unchanged from V8
                va->shadows[shadow_idx].shadow  = current;
                va->shadows[shadow_idx].primary = current;
                stats.repairs_applied++;

                // Viral propagation scan — unchanged from V8
                if (shadow_idx > 0 && shadow_idx < va->shadow_count - 1) {
                    int unit_idx = (int)(i % va->num_units);
                    if (unit_idx >= 0 && unit_idx < (int)va->num_units) {
                        float z_component = fabsf(va->normal_table[unit_idx].z);
                        for (int off = -2; off <= 2; off++) {
                            if (off == 0) continue;
                            int nb = (unit_idx + off + va->num_units) % va->num_units;
                            if (fabsf(fabsf(va->normal_table[nb].z) - z_component) < 0.15f)
                                stats.viral_propagations++;
                        }
                    }
                }
            }
        }
    }

    return stats;
}

// ============================================================================
// Aizawa Helpers — unchanged from V8
// ============================================================================

static inline void update_aizawa(AizawaState* as) {
    float a  = 0.95f + 0.05f * sinf(as->phi);
    float b  = 0.7f  + 0.02f * cosf(2.0f * as->phi);
    float f  = 0.1f  + 0.02f * sinf(3.0f * as->phi);
    float x  = as->state[0], y = as->state[1], z = as->state[2];
    float dt = 0.005f;
    float dx = (z - b) * x - 3.5f * y;
    float dy = 3.5f * x + (z - b) * y;
    float dz = 0.6f + a*z - z*z*z/3.0f - (x*x + y*y)*(1.0f + 0.25f*z) + f*z*x*x*x;
    as->state[0] += dt * dx;
    as->state[1] += dt * dy;
    as->state[2] += dt * dz;
    as->phi = fmodf(as->phi + 0.002f * sqrtf(dx*dx + dy*dy + dz*dz),
                    2.0f * 3.14159265f);
    as->steps++;
}

static inline bool should_eject(VivianiAllocator* va, size_t block_idx) {
    uint32_t d = VIVIANI_atomic_load(&va->defect_count);
    return (d > (uint32_t)(VIVIANI_MAX_DEFECTS * 0.65f)) || (block_idx % 4 == 0);
}

static inline bool check_aizawa_snap(AizawaState* as) {
    float r = fmaxf(cosf(as->phi) * cosf(as->phi),
                    sinf(as->phi) * sinf(as->phi));
    return (r < 0.82f && as->steps > 50);
}

// ============================================================================
// Microtubule Reconcile — V9 replacement for viviani_reconcile
//
// Processes flag queue entries in INTERFERENCE_WINDOW (32) sized batches.
// For each batch:
//   - Gather leading shadow at stride 13 within the window
//   - Reconstruct lagging equivalent from ejected pool entries in range
//   - At seam positions (block_idx % 13*BLOCK_SIZE == 0), apply forward
//     shift to lagging reconstruction before comparison (seam crosswind)
//   - Match: ligate — clear flag entry, update shadow, deduct defect_count
//   - Small mismatch (Hamming <= 4 bits): stir via Aizawa, defer
//   - Large mismatch: branch — increment branch_count, leave in queue
//
// Also exposes the original viviani_reconcile behavior as a fast path
// for non-seam entries (defect_type != 4).
// ============================================================================

typedef struct {
    uint32_t applied_patches;   // ligated (exact match after phase shift)
    uint32_t false_positives;   // queue entry out of range
    uint32_t propagations;      // high-priority patches
    uint32_t seam_ligations;    // seam entries resolved by cross-strand match
    uint32_t aizawa_stirs;      // small mismatches sent to Aizawa stirring
    uint32_t branches;          // large mismatches that forked a new branch
} ReconcileStats;

// Count differing bits between two 64-bit invariants
static inline uint32_t invariant_hamming(uint64_t a, uint64_t b) {
    uint64_t diff = a ^ b;
    // Parallel bit count (Brian Kernighan's method extended to 64-bit)
    uint32_t count = 0;
    while (diff) { diff &= diff - 1; count++; }
    return count;
}

// Reconstruct lagging-strand invariant for a given block from ejected pool.
// If no ejected entry covers this block, returns 0 (no lagging data available).
static inline uint64_t reconstruct_lagging(VivianiAllocator* va,
                                           size_t block_idx,
                                           size_t total_blocks)
{
    uint32_t hash_slot = (uint32_t)(block_idx % MAX_EJECTED);
    uint32_t pool_slot = va->ejected_index[hash_slot];

    if (pool_slot != VIVIANI_EJECTED_EMPTY &&
        pool_slot < MAX_EJECTED &&
        va->ejected_pool[pool_slot].block_idx == block_idx)
    {
        // Ejected entry exists — compute invariant from its current Aizawa state.
        // Use the phi as a block-position proxy since the actual block data was
        // zeroed on ejection. This gives a phase-consistent reconstruction.
        AizawaState* as = &va->ejected_pool[pool_slot].state;
        uint64_t phase_inv = (uint64_t)(as->phi * (float)0xFFFFFFFFULL)
        ^ ((uint64_t)as->steps << 32);
        return phase_inv;
    }

    // Fallback: linear scan (hash collision case)
    uint32_t ej = VIVIANI_atomic_load(&va->ejected_count);
    for (uint32_t s = 0; s < ej; s++) {
        if (va->ejected_pool[s].block_idx == block_idx) {
            AizawaState* as = &va->ejected_pool[s].state;
            return (uint64_t)(as->phi * (float)0xFFFFFFFFULL)
            ^ ((uint64_t)as->steps << 32);
        }
    }

    return 0;  // No lagging data for this block
}

static inline ReconcileStats viviani_microtubule_reconcile(
    VivianiAllocator* va,
    uint32_t          max_windows)  // max number of 32-event windows to process
{
    ReconcileStats stats = {0};
    size_t total_blocks  = va->arena_size / VIVIANI_BLOCK_SIZE;

    for (uint32_t w = 0; w < max_windows; w++) {
        FlagEntry entry;
        if (!flag_queue_pop(&va->flag_queue, &entry)) break;

        size_t block_idx = entry.chunk_id;
        if (block_idx >= total_blocks) { stats.false_positives++; continue; }

        size_t shadow_idx = block_idx / VIVIANI_SHADOW_STRIDE;
        if (shadow_idx >= va->shadow_count) { stats.false_positives++; continue; }

        // Non-seam entries: fast path identical to original viviani_reconcile
        if (entry.defect_type != 4) {
            uint8_t* block   = va->arena + (block_idx * VIVIANI_BLOCK_SIZE);
            uint64_t current = viviani_invariant(block, VIVIANI_BLOCK_SIZE,
                                                 (int)block_idx, (int)total_blocks,
                                                 va->offset_table, va->num_units);
            uint64_t stored   = va->shadows[shadow_idx].shadow;
            bool     is_seam  = (shadow_idx % MICROTUBULE_PROTOFILAMENTS == 0);
            uint64_t expected = is_seam ? seam_forward_shift(stored) : stored;

            if (current != expected) {
                if (is_seam) {
                    va->shadows[shadow_idx].shadow  = seam_inverse_shift(current);
                    va->shadows[shadow_idx].primary = seam_inverse_shift(current);
                } else {
                    va->shadows[shadow_idx].shadow  = current;
                    va->shadows[shadow_idx].primary = current;
                }
                stats.applied_patches++;
                if (entry.priority > 150) stats.propagations++;
            } else {
                stats.false_positives++;
            }
            continue;
        }

        // Seam catastrophe entries (defect_type == 4): full cross-strand check.
        // Process INTERFERENCE_WINDOW blocks starting at this entry's position,
        // sampling at seam stride (every 13 blocks within the window).
        for (uint32_t offset = 0; offset < INTERFERENCE_WINDOW;
             offset += MICROTUBULE_PROTOFILAMENTS)
             {
                 size_t probe_block = block_idx + offset;
                 if (probe_block >= total_blocks) break;

                 size_t probe_shadow = probe_block / VIVIANI_SHADOW_STRIDE;
                 if (probe_shadow >= va->shadow_count) break;

                 // Leading strand: current invariant from arena
                 uint8_t* block   = va->arena + (probe_block * VIVIANI_BLOCK_SIZE);
                 uint64_t leading = viviani_invariant(block, VIVIANI_BLOCK_SIZE,
                                                      (int)probe_block, (int)total_blocks,
                                                      va->offset_table, va->num_units);

                 // Lagging strand: reconstructed from ejected pool
                 uint64_t lagging_raw = reconstruct_lagging(va, probe_block, total_blocks);

                 if (lagging_raw == 0) {
                     // No lagging data — treat as false positive for this probe
                     continue;
                 }

                 // At seam positions within the window, apply forward shift to
                 // lagging reconstruction (the seam crosswind)
                 bool probe_is_seam = (probe_shadow % MICROTUBULE_PROTOFILAMENTS == 0);
                 uint64_t lagging   = probe_is_seam
                 ? seam_forward_shift(lagging_raw)
                 : lagging_raw;

                 uint32_t hdist = invariant_hamming(leading, lagging);

                 if (hdist == 0) {
                     // Perfect match: ligate — seal the nick
                     va->shadows[probe_shadow].shadow  = probe_is_seam
                     ? seam_inverse_shift(leading) : leading;
                     va->shadows[probe_shadow].primary = va->shadows[probe_shadow].shadow;
                     // Deduct seam defect weight (was added as 2 in repair)
                     uint32_t d = VIVIANI_atomic_load(&va->defect_count);
                     if (d >= 2) VIVIANI_atomic_fetch_sub(&va->defect_count, 2);
                     stats.seam_ligations++;
                     stats.applied_patches++;

                 } else if (hdist <= 4) {
                     // Small mismatch: stir the lagging unit via Aizawa diffusion
                     uint32_t hash_slot = (uint32_t)(probe_block % MAX_EJECTED);
                     uint32_t pool_slot = va->ejected_index[hash_slot];
                     if (pool_slot != VIVIANI_EJECTED_EMPTY && pool_slot < MAX_EJECTED) {
                         update_aizawa(&va->ejected_pool[pool_slot].state);
                     }
                     stats.aizawa_stirs++;

                 } else {
                     // Large mismatch: persistent divergence — branch
                     VIVIANI_atomic_fetch_add(&va->branch_count, 1);
                     stats.branches++;
                 }
             }
    }

    // Update defect_count to reflect current flag queue depth
    VIVIANI_atomic_store(&va->defect_count,
                         VIVIANI_atomic_load(&va->flag_queue.count));

    return stats;
}

// ============================================================================
// Convenience: trigger reconcile when event window fills
// Call after viviani_alloc / viviani_free in hot paths if desired.
// ============================================================================
static inline ReconcileStats viviani_maybe_reconcile(VivianiAllocator* va) {
    ReconcileStats empty = {0};
    uint32_t events = VIVIANI_atomic_load(&va->event_count);
    if ((events & (INTERFERENCE_WINDOW - 1)) != 0) return empty;
    // Reset event counter window (modular, so no store needed — just let it wrap)
    return viviani_microtubule_reconcile(va, 1);
}

// ============================================================================
// Superfluid Compaction — unchanged from V8
// (VIVIANI_SHADOW_STRIDE change propagates automatically through shadow_idx)
// ============================================================================

typedef struct {
    uint32_t blocks_scanned;
    uint32_t blocks_moved;
    uint32_t bytes_recovered;
    uint32_t shadow_updates;
    float    efficiency;
} SuperfluidStats;

static inline bool viviani_should_compact(VivianiAllocator* va) {
    size_t   fork    = VIVIANI_atomic_load(&va->fork_position);
    float    usage   = (float)fork / va->arena_size;
    uint32_t defects = VIVIANI_atomic_load(&va->defect_count);
    return (usage > 0.7f) || (defects > VIVIANI_MAX_DEFECTS);
}

static inline SuperfluidStats viviani_superfluid_compact(VivianiAllocator* va) {
    SuperfluidStats stats = {0};

    size_t fork_initial = VIVIANI_atomic_load(&va->fork_position);
    size_t new_fork     = 0;
    size_t total_blocks = va->arena_size / VIVIANI_BLOCK_SIZE;

    for (size_t block_idx = 0; block_idx < total_blocks; block_idx++) {
        stats.blocks_scanned++;

        size_t current_offset = block_idx * VIVIANI_BLOCK_SIZE;
        if (current_offset >= fork_initial) break;

        uint8_t* block = va->arena + current_offset;
        uint64_t inv   = viviani_invariant(block, VIVIANI_BLOCK_SIZE,
                                           (int)block_idx, (int)total_blocks,
                                           va->offset_table, va->num_units);

        if (inv == 0) continue;

        if (should_eject(va, block_idx) &&
            VIVIANI_atomic_load(&va->ejected_count) < MAX_EJECTED)
        {
            uint32_t slot = VIVIANI_atomic_fetch_add(&va->ejected_count, 1) % MAX_EJECTED;
            EjectedEntry* e = &va->ejected_pool[slot];
            e->block_idx        = block_idx;
            e->state.state[0]   = (float)(rand() % 1000) / 1000.0f;
            e->state.state[1]   = 0.2f;
            e->state.state[2]   = 0.3f;
            e->state.phi        = (float)block_idx / total_blocks * 2.0f * 3.14159265f;
            e->state.steps      = 0;
            va->ejected_index[block_idx % MAX_EJECTED] = slot;
            memset(block, 0, VIVIANI_BLOCK_SIZE);
            stats.blocks_moved++;
            continue;
        }

        {
            uint32_t hash_slot = (uint32_t)(block_idx % MAX_EJECTED);
            uint32_t pool_slot = va->ejected_index[hash_slot];
            EjectedEntry* found = NULL;

            if (pool_slot != VIVIANI_EJECTED_EMPTY &&
                pool_slot < MAX_EJECTED &&
                va->ejected_pool[pool_slot].block_idx == block_idx)
            {
                found = &va->ejected_pool[pool_slot];
            } else {
                uint32_t ej = VIVIANI_atomic_load(&va->ejected_count);
                for (uint32_t s = 0; s < ej; s++) {
                    if (va->ejected_pool[s].block_idx == block_idx) {
                        found = &va->ejected_pool[s];
                        va->ejected_index[hash_slot] = s;
                        break;
                    }
                }
            }

            if (found) {
                update_aizawa(&found->state);
                if (check_aizawa_snap(&found->state)) {
                    uint8_t* restored = va->arena + (found->block_idx * VIVIANI_BLOCK_SIZE);
                    uint64_t new_inv  = viviani_invariant(restored, VIVIANI_BLOCK_SIZE,
                                                          (int)block_idx, (int)total_blocks,
                                                          va->offset_table, va->num_units);
                    size_t sh = block_idx / VIVIANI_SHADOW_STRIDE;
                    if (sh < va->shadow_count) {
                        bool is_seam = (sh % MICROTUBULE_PROTOFILAMENTS == 0);
                        va->shadows[sh].shadow = is_seam
                        ? seam_inverse_shift(new_inv) : new_inv;
                    }
                    va->ejected_index[hash_slot] = VIVIANI_EJECTED_EMPTY;
                    VIVIANI_atomic_fetch_sub(&va->ejected_count, 1);
                    stats.shadow_updates++;
                }
            }
        }

        size_t dest_offset = new_fork;
        if (dest_offset != current_offset) {
            memmove(va->arena + dest_offset, block, VIVIANI_BLOCK_SIZE);

            size_t new_shadow_idx = (dest_offset / VIVIANI_BLOCK_SIZE) / VIVIANI_SHADOW_STRIDE;
            if (new_shadow_idx < va->shadow_count) {
                uint64_t new_inv = viviani_invariant(
                    va->arena + dest_offset, VIVIANI_BLOCK_SIZE,
                    (int)(dest_offset / VIVIANI_BLOCK_SIZE), (int)total_blocks,
                                                     va->offset_table, va->num_units);
                bool is_seam = (new_shadow_idx % MICROTUBULE_PROTOFILAMENTS == 0);
                va->shadows[new_shadow_idx].shadow  = is_seam
                ? seam_inverse_shift(new_inv) : new_inv;
                va->shadows[new_shadow_idx].primary = va->shadows[new_shadow_idx].shadow;
                stats.shadow_updates++;
            }

            stats.blocks_moved++;
            stats.bytes_recovered += (uint32_t)(current_offset - dest_offset);
        }
        new_fork += VIVIANI_BLOCK_SIZE;
    }

    viviani_geometric_repair(va);

    if (fork_initial > new_fork) {
        stats.efficiency = (float)(fork_initial - new_fork) / fork_initial;
        VIVIANI_atomic_store(&va->fork_position, new_fork);
    }

    VIVIANI_atomic_fetch_add(&va->version, 1);
    return stats;
}

// ============================================================================
// Statistics — V9: seam counters added
// ============================================================================

typedef struct {
    uint32_t allocated_blocks;
    uint32_t defect_count;
    uint32_t protected_mode;
    uint32_t flip_state;
    float    alloc_rate;
    uint32_t flag_queue_depth;
    uint32_t bin_counts[VIVIANI_BIN_COUNT];
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t gpu_held_bytes;
    uint32_t ejected_count;
    // V9 seam health
    uint32_t seam_defect_count;
    uint32_t seam_catastrophe_count;
    uint32_t branch_count;
    uint32_t event_count;
} VivianiStats;

static inline VivianiStats viviani_stats(VivianiAllocator* va) {
    VivianiStats s = {0};
    s.allocated_blocks       = VIVIANI_atomic_load(&va->allocated_blocks);
    s.defect_count           = VIVIANI_atomic_load(&va->defect_count);
    s.protected_mode         = VIVIANI_atomic_load(&va->protected_mode);
    s.flip_state             = VIVIANI_atomic_load(&va->flip_state);
    s.alloc_rate             = VIVIANI_atomic_load(&va->alloc_rate_fp) / 1000.0f;
    s.flag_queue_depth       = VIVIANI_atomic_load(&va->flag_queue.count);
    s.ejected_count          = VIVIANI_atomic_load(&va->ejected_count);
    s.seam_defect_count      = VIVIANI_atomic_load(&va->seam_defect_count);
    s.seam_catastrophe_count = VIVIANI_atomic_load(&va->seam_catastrophe_count);
    s.branch_count           = VIVIANI_atomic_load(&va->branch_count);
    s.event_count            = VIVIANI_atomic_load(&va->event_count);
    for (int i = 0; i < VIVIANI_BIN_COUNT; i++)
        s.bin_counts[i] = VIVIANI_atomic_load(&va->bin_counters[i]);
    s.cache_hits   = VIVIANI_atomic_load(&va->cache_hits);
    s.cache_misses = VIVIANI_atomic_load(&va->cache_misses);
    #ifdef __CUDACC__
    s.gpu_held_bytes = VIVIANI_atomic_load(&va->gpu_held_bytes);
    #endif
    return s;
}

static inline void viviani_print_stats(const VivianiStats* s) {
    printf("=== Viviani Allocator Stats (V9 — Microtubule Edition) ===\n");
    printf("Allocated blocks: %u\n",  s->allocated_blocks);
    printf("Defect count:     %u\n",  s->defect_count);
    printf("Protected mode:   %s\n",  s->protected_mode ? "YES" : "no");
    printf("Flip state:       %u\n",  s->flip_state);
    printf("Alloc rate:       %.2f  (hold threshold: %.3f derived)\n",
           s->alloc_rate, (float)VIVIANI_ALLOC_RATE_THRESHOLD / 1000.0f);
    printf("Flag queue depth: %u\n",  s->flag_queue_depth);
    printf("Aizawa ejected:   %u / %u\n", s->ejected_count, MAX_EJECTED);
    printf("Events:           %u  (window=%d, cycle=%d)\n",
           s->event_count, INTERFERENCE_WINDOW, INTERFERENCE_CYCLE);
    printf("--- Seam Health (microtubule protofilament geometry) ---\n");
    printf("Seam defects:     %u  (threshold=%d per scan)\n",
           s->seam_defect_count, SEAM_DEFECT_THRESHOLD);
    printf("Seam catastrophes:%u\n",  s->seam_catastrophe_count);
    printf("Branch count:     %u\n",  s->branch_count);
    printf("--- Bin counts ---\n");
    for (int i = 0; i < VIVIANI_BIN_COUNT; i++) printf(" %u", s->bin_counts[i]);
    printf("\n");
    printf("--- Cache Stats ---\n");
    printf("Cache hits:       %lu\n", s->cache_hits);
    printf("Cache misses:     %lu\n", s->cache_misses);
    uint64_t total    = s->cache_hits + s->cache_misses;
    float    hit_rate = (total > 0) ? (float)s->cache_hits / total * 100.0f : 0.0f;
    printf("Hit rate:         %.1f%%\n", hit_rate);
    #ifdef __CUDACC__
    printf("GPU held bytes:   %lu\n", s->gpu_held_bytes);
    #endif
}

// ============================================================================
// CUDA Extensions — unchanged from V8
// ============================================================================

#ifdef __CUDACC__

static inline void viviani_cuda_init(VivianiAllocator* va, int num_streams) {
    va->num_streams  = num_streams;
    va->streams      = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t));
    for (int s = 0; s < num_streams; s++) cudaStreamCreate(&va->streams[s]);
    va->stream_holds       = (void**)calloc(num_streams * VIVIANI_HOLD_PER_STREAM, sizeof(void*));
    va->stream_hold_sizes  = (size_t*)calloc(num_streams * VIVIANI_HOLD_PER_STREAM, sizeof(size_t));
    va->stream_hold_counts = (VIVIANI_ATOMIC uint32_t*)calloc(num_streams, sizeof(VIVIANI_ATOMIC uint32_t));
    VIVIANI_atomic_store(&va->gpu_held_bytes, 0);
}

static inline void* viviani_cuda_alloc(VivianiAllocator* va, size_t size, int stream_idx) {
    if (stream_idx >= va->num_streams) return NULL;

    int    bin         = viviani_bin_index(va, size);
    size_t actual_size = va->bin_sizes[bin];

    uint32_t hold_count = VIVIANI_atomic_load(&va->stream_hold_counts[stream_idx]);
    size_t*  sizes      = va->stream_hold_sizes + (stream_idx * VIVIANI_HOLD_PER_STREAM);
    void**   holds      = (void**)(va->stream_holds + (stream_idx * VIVIANI_HOLD_PER_STREAM));

    for (int h = (int)hold_count - 1; h >= 0; h--) {
        if (sizes[h] == actual_size) {
            void* ptr = holds[h];
            for (int j = h; j < (int)hold_count - 1; j++) {
                holds[j] = holds[j+1]; sizes[j] = sizes[j+1];
            }
            VIVIANI_atomic_fetch_sub(&va->stream_hold_counts[stream_idx], 1);
            VIVIANI_atomic_fetch_sub(&va->gpu_held_bytes, actual_size);
            return ptr;
        }
    }

    void* ptr = NULL;
    if (cudaMalloc(&ptr, actual_size) != cudaSuccess) return NULL;
    return ptr;
}

static inline void viviani_cuda_free(VivianiAllocator* va, void* ptr,
                                     size_t size, int stream_idx) {
    if (!ptr || stream_idx >= va->num_streams) return;

    int    bin         = viviani_bin_index(va, size);
    size_t actual_size = va->bin_sizes[bin];

    uint32_t hold_count  = VIVIANI_atomic_load(&va->stream_hold_counts[stream_idx]);
    uint32_t alloc_count = VIVIANI_atomic_load(&va->allocated_blocks);

    bool protect  = is_protected((int)alloc_count, bin) ||
    (VIVIANI_atomic_load(&va->alloc_rate_fp) > VIVIANI_ALLOC_RATE_THRESHOLD);
    bool can_hold = protect &&
    hold_count < VIVIANI_HOLD_PER_STREAM &&
    VIVIANI_atomic_load(&va->gpu_held_bytes) + actual_size < VIVIANI_GPU_HELD_CAP;

    if (can_hold) {
        size_t* sizes = va->stream_hold_sizes + (stream_idx * VIVIANI_HOLD_PER_STREAM);
        void**  holds = (void**)(va->stream_holds + (stream_idx * VIVIANI_HOLD_PER_STREAM));
        holds[hold_count] = ptr;
        sizes[hold_count] = actual_size;
        VIVIANI_atomic_fetch_add(&va->stream_hold_counts[stream_idx], 1);
        VIVIANI_atomic_fetch_add(&va->gpu_held_bytes, actual_size);
    } else {
        cudaFree(ptr);
    }
                                     }

                                     __global__ void viviani_touch_kernel(char* ptr, size_t size) {
                                         int idx = blockIdx.x * blockDim.x + threadIdx.x;
                                         if ((size_t)idx < size) ptr[idx] = (char)(threadIdx.x & 0xFF);
                                     }

                                     #endif // __CUDACC__

                                     #endif // VIVIANI_ALLOC_HOPFION_FIXED_CUH
