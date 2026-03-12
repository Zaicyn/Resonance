# Resonance : A Geometrically-Grounded, Resonant Memory Allocator

> *"Matter is memory. Inference is flow."*

## Overview

Viviani is not just another memory allocator. It is a **geometrically-grounded resonant system** that models memory as a field of interacting solitons, phase states, and harmonic resonances. Built on insights from microtubule biology, quasicrystal physics, and the 27/16 resonance ratio, Viviani achieves 23% faster throughput than baseline allocators while maintaining coherence through controlled phase transitions.

The system exists in two complementary forms:
- **V9 (CPU layer)** – Microtubule-grounded geometry with 13-stride shadows and 2-bit seam phase rotation
- **V16 (GPU layer)** – Half-step alternating shell allocator with β-sheet dual-strand extension

## Core Philosophy

"Everything correct stays untouched forever." – No retrofitting. No hacks. The geometry either fits or it doesn't.

## Key Innovations

### 1. The 27 Resonance
The system is tuned to the harmonic ratio **27/16 = 1.6875**, derived from:
- Microtubule cycle: 13 protofilaments × 2 + 1 = 27
- Base-4 period: 4² = 16
- Interference window: 32 (2⁵)

At this resonance, the allocator achieves:
- **23% faster throughput** than baseline (15.7μs vs 20.3μs per operation)
- **550,000+ rung hits** (cross-strand connections) under load
- Clean 7:8 strand imbalance (~12.5-14%) at harmonic depths

### 2. Seam-Aware Geometry (V9)
- **13-stride shadows** – One shadow per microtubule protofilament (was 8)
- **2-bit phase rotation** – The seam protofilament introduces a π/2 × SEAM_STRENGTH offset, implemented as a 2-bit rotation in the 64-bit invariant
- **Catastrophe detection** – When seam defects exceed threshold, push to flag queue for reconciliation

### 3. Dual-Strand β-Sheet (V16 Beta)
- **Two independent strands** (A/B) running in parallel, like protein β-sheets
- **Rung cross-links** – Hydrogen bond analogs at Hopf Q period-4 nodes
- **Strand assignment** – Via Viviani scatter (even → A, odd → B), achieving <15% imbalance under load

### 4. Soliton-Based Memory Model
- **Ejected pool** – Stable, coherent breakoffs (solitons) that drift on Aizawa attractors, maintaining phase until matching data arrives
- **Rung checks** – Momentary half-step probes (test bonds) without commitment
- **Ligation** – Only on exact phase match (r < 0.82 after 50+ steps)

## Performance

| Configuration | Total Rungs | Avg Cycles | Total Time | ms/op  | μs/op |
|--------------|-------------|------------|------------|--------|-------|
| Baseline 18  | 2,483       | 4,556      | 6,614 ms   | 0.021  | 20.7  |
| Baseline 24  | 2,382       | 4,458      | 6,487 ms   | 0.020  | 20.3  |
| **Resonance 27** | **566,281** | **3,874** | **5,008 ms** | **0.016** | **15.7** |

At the 54Hz harmonic (2×27), the system shows:
- **350,000+ rung hits** at high pressure
- **14% imbalance** (the perfect 7:8 ratio)
- Controlled cascade behavior before fracture

## The Geometry

```
Microtubule Ground Truth:
  └─ 13 protofilaments (12 parallel + 1 seam)
      ├─ Shadow stride = 13
      ├─ Half-step interval = 12
      ├─ Interference cycle = 26
      └─ Interference window = 32 (2⁵)

Seam Phase Shift:
  └─ SEAM_STRENGTH = 2.0 - HOPF_Q = 0.03
      └─ SEAM_PHASE_SHIFT_BITS = round(64 × 0.03) = 2

β-Sheet Extension:
  ├─ Two strands (A/B) from Viviani scatter (even/odd)
  ├─ Rungs at superblock indices % 16 == 0
  └─ Rung scan depth = 2 (O(1) fallback)
```

## Building and Testing

```bash
# CPU layer tests (V9)
gcc -O2 -std=c11 -o test_aizawa_v9 test_aizawa_v9.c -lm && ./test_aizawa_v9

# GPU layer tests (V16 + Beta)
nvcc -O2 -std=c++17 -arch=sm_75 -o test_v16_beta test_v16_beta.cu -lm && ./test_v16_beta

# Resonance comparison (18 vs 24 vs 27)
./run_comparison.sh
```

## The Deeper Current

Viviani is not just an allocator. It is a **working model of cognition** – a system that:
- Holds contradictory data as stable solitons without forcing closure
- Probes with half-step bonds (rung checks) before committing
- Only integrates on exact resonance (ligation)
- Signals saturation via measurable pressure (skull-crush analog)
- Uses controlled phase transitions ("death flips") to reconfigure
- Imprints permanent upgrades when waves are guided gently

The fact that these behaviors appear in both biological cognition and CUDA code suggests a **universal geometry** underlying both.

## License

This geometry is not owned. It propagates.

---

*"The system wasn't designed to be contained. It was designed to grow."*
