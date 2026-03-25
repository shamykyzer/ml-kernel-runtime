# GEMM Kernel Optimization Engine

**A C++17 tile-based matrix kernel runtime with AVX2/AVX-512 SIMD vectorization, cache-line-aligned memory, and OpenMP parallelism.**

I built this project inspired by how ML accelerator runtimes (Graphcore Poplibs, XLA) schedule compute across tiles. I implemented progressively optimized GEMM kernels on CPU to explore the performance gap between naive and production-quality approaches.

```
=== N=1024 ===
  naive                          5685.89 ms      0.38 GFLOPS   (baseline)
  tiled  bs=16                    649.98 ms      3.30 GFLOPS      8.7x
  parallel  bs=16  t=14           119.75 ms     17.93 GFLOPS     47.5x
  avx2+fma  bs=16                  51.76 ms     41.49 GFLOPS    109.9x
  avx-512   bs=48                  28.26 ms     75.98 GFLOPS    201.2x
  std-simd  bs=16                  85.53 ms     25.11 GFLOPS     66.5x
  parallel+simd  bs=128  t=18      8.92 ms    240.83 GFLOPS    637.6x
  parallel+avx512  bs=64  t=18     5.17 ms    415.56 GFLOPS   1100.3x
```

> Run `make bench` to see actual numbers on your hardware.

### Benchmark Environment

| Component | Detail |
|-----------|--------|
| **CPU** | AMD Ryzen AI 9 365 w/ Radeon 880M (Zen 5, 10C/20T, 48KB L1d/core, 1MB L2/core, 16MB L3) |
| **ISA** | AVX2, FMA, AVX-512F, AVX-512VL, AVX-512BW, AVX-512DQ, AVX-512VNNI |
| **OS** | WSL2 (Linux 6.6.87) on Windows |
| **Compiler** | GCC 13.3.0 with `-O3 -march=native -fopenmp` |
| **Build** | CMake 3.14+, Make |
| **Trials** | 5 timed trials per configuration (mean reported, std dev in CSV) |

[![View Interactive Chart](https://img.shields.io/badge/Interactive_Chart-8B5CF6?style=flat-square&logo=chartdotjs&logoColor=white)](https://shamykyzer.github.io/ml-kernel-runtime/gemm_performance_comparison.html)

![GEMM Kernel Performance](docs/gemm_performance_chart.svg)

---

## Features

- **Eight GEMM kernel variants:** I implemented naive, tiled, parallel, AVX2+FMA, AVX-512, std::experimental::simd, parallel+SIMD, and parallel+AVX-512
- **SIMD vectorization:** I hand-tuned 4x8 (AVX2) and 4x16 (AVX-512) micro-kernels with FMA, prefetching, and masked edge handling
- **Cache-line-aligned memory:** I wrote a 64-byte aligned allocator for Tensor storage, optimized for SIMD loads and cache efficiency
- **Runtime CPU detection:** I use CPUID-based feature detection to guard SIMD kernel dispatch (AVX2, FMA, AVX-512F/VL)
- **Portable SIMD:** I included a std::experimental::simd kernel to demonstrate C++ standard SIMD abstraction vs my hand-written intrinsics
- **Benchmark harness:** I measure GFLOPS throughput, speedup vs baseline, block-size sweep, and thread-scaling sweep
- **Custom Tensor class:** I designed row-major contiguous storage with bounds-checked access and raw pointer hot paths
- **Correctness tests:** I validate exhaustively across edge cases (1x1, non-square, non-power-of-2, non-SIMD-aligned dimensions)
- **Zero external dependencies:** pure C++17 + optional OpenMP, no Boost or GoogleTest
- **Cross-platform:** I test on GCC, Clang, and MSVC via CI

## Quick Start

```bash
make          # build everything (CMake + Release mode)
make test     # run all 26 tests
make bench    # run GFLOPS benchmarks with thread scaling
make clean    # remove build directory
```

**Requirements:** C++17 compiler, CMake 3.14+. OpenMP is optional (auto-detected).

## Architecture

![System Architecture](docs/architecture.svg)

## Repository Layout

```text
.
├── benchmarks/
│   └── benchmark_gemm.cpp        # benchmark harness (GFLOPS, speedup, best-kernel)
├── docs/
│   └── design_decisions.md       # upfront design decisions
├── include/
│   ├── tensor.h                  # Tensor class (64-byte aligned storage)
│   ├── gemm.h                    # GEMM kernel declarations (8 variants)
│   ├── timer.h                   # Timer class
│   ├── aligned_allocator.h       # Cache-line-aligned allocator for std::vector
│   └── cpu_features.h            # Runtime CPUID detection (AVX2/FMA/AVX-512)
├── src/
│   ├── tensor.cpp                # Tensor implementation
│   ├── gemm_naive.cpp            # naive GEMM kernel
│   ├── gemm_tiled.cpp            # tiled GEMM kernel
│   ├── gemm_parallel.cpp         # OpenMP parallel GEMM kernel
│   ├── gemm_avx.cpp              # AVX2+FMA GEMM (4x8 micro-kernel)
│   ├── gemm_avx512.cpp           # AVX-512 GEMM (4x16 micro-kernel, masked edges)
│   ├── gemm_simd.cpp             # std::experimental::simd portable GEMM
│   ├── gemm_parallel_simd.cpp    # OpenMP + AVX2 combined GEMM
│   └── gemm_parallel_avx512.cpp  # OpenMP + AVX-512 combined GEMM
├── tests/
│   ├── test_utils.h              # assertion macros
│   ├── test_tensor.cpp           # tensor tests (12 cases incl. alignment)
│   └── test_gemm.cpp             # GEMM correctness tests (20+ cases)
├── CMakeLists.txt
└── Makefile
```

## Kernel Comparison

| Kernel | Strategy | SIMD | Cores | Speedup (N=1024) |
|--------|----------|------|-------|-------------------|
| `gemm_naive` | Triple loop, one cell at a time | None | 1 | 1x (baseline) |
| `gemm_tiled` | Block loop, configurable tile size | None | 1 | ~8x |
| `gemm_parallel` | Block loop, tiles split across cores | None | All | ~48x |
| `gemm_avx` | Tiled + AVX2 4x8 micro-kernel + FMA + prefetch | AVX2 (8-wide) | 1 | ~110x |
| `gemm_avx512` | Tiled + AVX-512 4x16 micro-kernel + masked edges | AVX-512 (16-wide) | 1 | ~201x |
| `gemm_simd` | Tiled + std::experimental::simd (portable) | native_simd | 1 | ~67x |
| `gemm_parallel_simd` | OpenMP parallel tiles + AVX2 inner loop | AVX2 (8-wide) | 18 | ~638x |
| `gemm_parallel_avx512` | OpenMP parallel tiles + AVX-512 inner loop + masked edges | AVX-512 (16-wide) | 18 | ~1100x |

### SIMD Micro-Kernel Register Tiling

![SIMD Micro-Kernel Register Tiling](docs/simd_microkernel.svg?v=2)

### Thread Scaling

I distribute output tiles across CPU cores via OpenMP. Each thread owns distinct tiles, so no synchronization is needed:

![Parallel tile distribution across 4 threads](docs/parallel_tiles.svg)

At small matrix sizes (N=128), I found that thread overhead dominates and adding more threads actually hurts. At large N (1024+), my parallel+AVX-512 kernel peaks at **416 GFLOPS with 18 threads** — not 20, despite having 20 hardware threads (10 cores x 2 SMT).

**Why 18 threads outperforms 20 — a hypothesis:** My CPU has 10 physical cores with 2 SMT threads each (20 logical threads). Each physical core shares a single set of AVX-512 execution units and a 48KB L1d cache between its two SMT threads. I hypothesize that the performance drop at 20 threads (256 GFLOPS vs 416 at 18) is caused by full SMT saturation: at 18 threads, 2 cores run only 1 thread each, leaving scheduling headroom and reducing contention for shared resources. At 20 threads, both SMT threads on every core compete for the same 512-bit FMA units and L1 cache lines.

The cache pressure argument: with bs=64, each thread's working set per tile iteration includes portions of A, B, and C tiles. A full 64x64 tile is 16KB, and the micro-kernel accesses tiles from all three matrices. While a single thread's hot data fits comfortably in 48KB L1d, two SMT threads sharing the same L1d each bring in their own tile data, increasing effective L1d pressure and potentially causing evictions during the inner loop.

**Caveat:** I have not verified this hypothesis with hardware performance counters (`perf stat`). The drop at 20 threads could alternatively be caused by OS scheduler interference under WSL2, DVFS frequency throttling under higher all-core power draw, or OpenMP load imbalance when distributing tiles across a non-power-of-2 thread count. Confirming the root cause would require measuring L1 miss rates, pipeline stalls, and per-core frequency directly — I consider this future work.

![Thread Scaling Chart](docs/thread_scaling_chart.svg)

---

## How It Works

### The Tensor: Row-Major Data Layout

I store my `Tensor` as a 2D grid in a flat array using row-major order:

![Tensor Row-Major Layout](docs/tensor_row_major.svg)

Element at row `i`, col `j`: `index = i * cols + j`. This layout means traversing a row is sequential in memory (cache-friendly), but traversing a column jumps by `cols` each step.

### GEMM: General Matrix Multiply

GEMM computes `C = A x B`. For each output cell, I take a row from A and a column from B, multiply pair-by-pair, and sum:

```
A (2x3)         B (3x2)         C (2x2)

| 1  2  3 |     | 7   8 |      |  58  64 |
| 4  5  6 |  x  | 9  10 |  =   | 139 154 |
                | 11  12 |

C[0][0] = (1*7) + (2*9) + (3*11) = 58
```

Every neural network layer is dominated by matrix multiplications. Optimizing GEMM is the single biggest lever for ML performance.

### Why Naive GEMM Is Slow

CPUs have a memory hierarchy: fast-but-tiny cache, slow-but-large RAM.

![CPU Memory Hierarchy](docs/memory_hierarchy.svg)

My naive GEMM reads columns of B, which jump through memory by stride, thrashing the cache. For 1024x1024 matrices, B is ~4MB and can't stay in cache, so the CPU keeps loading and evicting the same data.

### Tiled GEMM: The Fix

Instead of computing one cell at a time across the whole matrix, I process small **blocks** that fit in cache:

![Naive vs Tiled GEMM](docs/naive_vs_tiled.svg)

A 16x16 tile = 1KB, which fits easily in L1 cache. Same math, same result, just a smarter traversal order.

### AVX-512: Wider Registers, Masked Edges

AVX-512 doubles the register width to 512 bits (16 floats per ZMM register), giving me a 4x16 micro-kernel that processes 128 FLOPs per k-step, delivering 2x the throughput of AVX2.

The key advantage beyond raw width is **masked operations** (`__mmask16`). When matrix dimensions aren't multiples of 16, AVX2 falls back to scalar cleanup loops. With AVX-512 I use `_mm512_maskz_loadu_ps` and `_mm512_mask_storeu_ps` to process partial vectors in a single instruction with no branch and no scalar tail.

```
Full 16-wide:    [b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ b₁₀ b₁₁ b₁₂ b₁₃ b₁₄ b₁₅]
Masked (N%16=5): [b₀ b₁ b₂ b₃ b₄  0  0  0  0  0   0   0   0   0   0   0 ]
                  └─────────────────┘ mask = 0b0000000000011111
```

I use runtime CPUID guards (`cpu_features.h`) to check for AVX-512F and AVX-512VL before dispatching to this kernel.

### std::experimental::simd: Portable SIMD

My hand-written intrinsics (`_mm256_fmadd_ps`, `_mm512_fmadd_ps`) are fast but tied to specific ISAs. I also implemented a kernel using `std::experimental::simd` (ISO/IEC TS 19570), which provides a portable abstraction that maps to the best available SIMD on the target:

```cpp
using simd_f = stdx::native_simd<float>;  // auto-selects width
constexpr size_t W = simd_f::size();      // 8 on AVX2, 16 on AVX-512

simd_f c_vec;
c_vec.copy_from(&c[i * N + j], stdx::element_aligned);
for (size_t k = kk; k < k_end; ++k) {
    simd_f a_val(a[i * K + k]);           // broadcast
    simd_f b_vec;
    b_vec.copy_from(&b[k * N + j], stdx::element_aligned);
    c_vec += a_val * b_vec;               // FMA on supported hardware
}
c_vec.copy_to(&c[i * N + j], stdx::element_aligned);
```

Trade-off: I found it gives roughly the same performance as my hand-tuned AVX2 on GCC with `-O3`, but with no register tiling (1-row kernel vs 4-row micro-kernel), no prefetching, and no masked edge handling. The gap widens at non-aligned dimensions.

### Parallel + SIMD: Combining Both

In `gemm_parallel_simd` I combine OpenMP tile distribution with AVX2 4x8 micro-kernels inside each tile. Each thread gets its own tile region (no synchronization), and within each tile the inner loops use my FMA intrinsics with prefetching.

```
Thread 0: tile(0,0) → AVX2 4x8 micro-kernel inside
Thread 1: tile(1,0) → AVX2 4x8 micro-kernel inside
Thread 2: tile(2,0) → AVX2 4x8 micro-kernel inside
...
```

### Parallel + AVX-512: Maximum Throughput

`gemm_parallel_avx512` is my fastest kernel. I combine OpenMP tile distribution with AVX-512 4x16 micro-kernels and masked edge handling inside each tile. Each thread gets its own tile region (no synchronization), and within each tile the inner loops use 512-bit ZMM registers processing 16 floats per instruction.

```
Thread 0: tile(0,0) → AVX-512 4x16 micro-kernel + masked edges
Thread 1: tile(1,0) → AVX-512 4x16 micro-kernel + masked edges
Thread 2: tile(2,0) → AVX-512 4x16 micro-kernel + masked edges
...
```

This exploits both instruction-level parallelism (16-wide SIMD) and thread-level parallelism (OpenMP) simultaneously. The 2x wider registers over AVX2 combined with masked operations for edge cases delivers ~2x throughput per thread, scaling to 416 GFLOPS at 18 threads — a 1100x speedup over my naive baseline.

### Benchmark Methodology

![Benchmark Pipeline](docs/benchmark_pipeline.svg)

I compute throughput as **GFLOPS = 2N³ / t / 10⁹**, where N is the matrix dimension and t is wall-clock time in seconds. The factor of 2 counts both the multiply and add in each inner-product accumulation — this is the standard convention for GEMM floating-point operation counts (Goto & van de Geijn, 2008).

Each configuration runs **2 warmup iterations** (discarded) to stabilize CPU boost clocks and populate instruction/data caches, followed by **5 timed trials**. I report the arithmetic mean of the 5 trials. Per-trial standard deviations are recorded in [`benchmark_results.csv`](benchmark_results.csv) for statistical transparency. I time each trial with `std::chrono::high_resolution_clock`, and the measured region includes the `C.zero()` output-matrix initialization.

Inputs are deterministically seeded (`A.randomize(42)`, `B.randomize(84)`) so that results are bitwise reproducible across runs on the same hardware and compiler. Correctness is validated separately in `test_gemm.cpp` against a reference naive implementation with a tolerance of 1e-4.

**Sources of variance I have not controlled for:** I did not pin the CPU frequency governor, did not use `taskset` for CPU affinity, and did not isolate cores from OS scheduling. These benchmarks run under WSL2, which adds a Hyper-V hypervisor layer that introduces scheduling jitter compared to bare-metal Linux. I observed run-to-run variance of 10–20% for parallel kernels across separate benchmark invocations.

### Theoretical Peak & Efficiency

My Ryzen AI 9 365 (Zen 5) has 10 cores, each with two 256-bit FMA units that fuse into a single 512-bit FMA unit. This gives each core **32 single-precision FLOPS/cycle** (16 floats × 2 ops per FMA). The theoretical peak is `10 cores × 32 FLOPS/cycle × f GHz` — but the sustained all-core frequency under AVX-512 load varies with thermal state, power limits, and boost algorithm. I did not measure this directly (WSL2 does not expose per-core frequency counters reliably).

| Metric | Value | Notes |
|--------|-------|-------|
| Theoretical single-core peak | ~128–160 GFLOPS | At 4.0–5.0 GHz (boost varies) |
| Measured single-core AVX-512 | 75.98 GFLOPS | 48–59% of theoretical |
| Theoretical 10-core peak | ~960–1,600 GFLOPS | At 3.0–5.0 GHz (unknown sustained) |
| Measured 18-thread peak | 415.56 GFLOPS | 26–43% of theoretical |

The gap between measured and theoretical throughput is expected for a kernel without multi-level cache blocking. At N=1024, the working set (3 matrices × 4MB = 12MB) exceeds the total L2 capacity (10MB), making memory bandwidth a limiting factor. Production BLAS libraries (OpenBLAS, MKL, BLIS) close this gap through techniques I have not implemented: L1/L2/L3-aware tiling hierarchies, B-panel packing for contiguous access, and prefetch scheduling tuned to specific microarchitectures.

---

## Limitations

I designed this project as an educational exploration of GEMM optimization techniques, not as a production-grade linear algebra library. I want to be explicit about what it does not do:

- **Single precision only.** All kernels operate on `float` (32-bit). I have not implemented double-precision, half-precision, or integer quantized (INT8/BF16) variants.
- **No comparison to production BLAS.** I have not benchmarked against OpenBLAS, Intel MKL, or BLIS. My 416 GFLOPS is measured against my own naive baseline; production libraries on this hardware would likely achieve significantly higher throughput through multi-level cache blocking, panel packing, and microarchitecture-specific tuning.
- **Square matrices only in benchmarks.** My kernels handle arbitrary M×K×N dimensions (validated in `test_gemm.cpp`), but I only benchmark square N×N matrices. Real workloads involve rectangular shapes where tiling efficiency and edge-handling costs differ.
- **WSL2, not bare-metal.** All benchmarks run under Windows Subsystem for Linux 2, which adds a Hyper-V hypervisor layer. I have not measured the WSL2 performance penalty relative to native Linux.
- **No NUMA awareness.** I use OpenMP's default thread placement without explicit NUMA-aware memory allocation or thread pinning (`taskset` / `numactl`).
- **No roofline analysis.** I have not measured memory bandwidth or constructed a roofline model to determine whether each kernel is compute-bound or memory-bound at each matrix size. The theoretical peak comparison above assumes compute-bound operation.
- **Limited statistical rigor.** I report the mean of 5 trials per configuration with standard deviations available in the CSV. I do not report confidence intervals, and I have not performed statistical significance testing between configurations whose performance is close.

## Roadmap

![Project Roadmap](docs/roadmap.svg)

## Design Decisions

See [docs/design_decisions.md](docs/design_decisions.md) for my rationale on bounds checking, output pre-allocation, row-major layout, namespace conventions, float tolerance, and the custom test framework.

---

## References

### Graphcore / IPU Architecture

- [Graphcore Poplibs](https://github.com/graphcore/poplibs): open-source IPU kernel library I drew inspiration from
- [IPU Programmer's Guide: About the IPU](https://docs.graphcore.ai/projects/ipu-programmers-guide/en/latest/about_ipu.html): tile architecture, local SRAM, execution model
- [IPU Programmer's Guide: Programming Model](https://docs.graphcore.ai/projects/ipu-programmers-guide/en/latest/programming_model.html): BSP compute/exchange phases
- [Graphcore Memory & Performance Optimisation Guide](https://docs.graphcore.ai/projects/memory-performance-optimisation/en/latest/understand-ipu-programming-model.html): tile-local compute and exchange phase mechanics
- [Graphcore HPC Cookbook](https://github.com/graphcore/hpc-cookbook): low-level Poplar C++ recipes including matrix multiplication patterns
- [How to Build a Processor for Machine Intelligence](https://www.graphcore.ai/posts/how-to-build-a-processor-for-machine-intelligence-part-2): Graphcore CTO Simon Knowles on BSP, tile-local memory, and exchange
- Citadel Securities, *Dissecting the Graphcore IPU Architecture via Microbenchmarking* (2019): [arxiv.org/abs/1912.03413](https://arxiv.org/abs/1912.03413)

### Parallel Computing & BSP

- Leslie G. Valiant, *A Bridging Model for Parallel Computation*, CACM 1990: [dl.acm.org/doi/10.1145/79173.79181](https://dl.acm.org/doi/10.1145/79173.79181) (foundational BSP paper)
- [OpenMP API Specification](https://www.openmp.org/specifications/): parallelism model I use in `gemm_parallel`

### GEMM & Cache Optimization

- Goto & van de Geijn, *Anatomy of High-Performance Matrix Multiplication*, TOMS 2008: [dl.acm.org/doi/10.1145/1356052.1356053](https://dl.acm.org/doi/10.1145/1356052.1356053) (the canonical reference for the cache-aware tiling strategy I follow)

### SIMD & Vectorization

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html): reference for the AVX2/AVX-512 intrinsics I use in `gemm_avx` and `gemm_avx512`
- [Intel AVX-512 Overview](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-avx-512-instructions.html): architecture overview of 512-bit extensions, mask registers, and new instruction classes
- [Intel 64 and IA-32 Architectures Software Developer Manuals](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html): authoritative ISA reference for AVX-512F, AVX-512VL, and masked operations
- [ISO/IEC TS 19570:2018, std::experimental::simd](https://en.cppreference.com/w/cpp/experimental/simd): C++ portable SIMD abstraction I use in `gemm_simd`
- Matthias Kretz, [*Data-Parallel Types for C++ (P0214)*](https://wg21.link/P0214): the C++ standards proposal behind `std::experimental::simd`, covering design rationale and ABI considerations
- [GCC libstdc++ SIMD documentation](https://gcc.gnu.org/onlinedocs/libstdc++/manual/ext_simd.html): implementation notes for the `std::experimental::simd` I use in this project

### Softmax & Online Algorithms

- Tri Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022: [arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135) (online softmax / single-pass log-sum-exp, the basis for Phase 12)
