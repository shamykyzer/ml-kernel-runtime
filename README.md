# Tile Based ML Kernel Runtime

A C++17 systems project that implements and benchmarks tile-based matrix kernels
with simple runtime-style scheduling, inspired by ML kernel runtimes such as
Graphcore Poplibs.

```mermaid
graph TD
    subgraph "Benchmark Layer"
        BENCH["benchmark_gemm
        (GFLOPS, speedup, best-kernel)"]
    end
    subgraph "Kernel Layer"
        direction LR
        NAIVE["gemm_naive"] ~~~ TILED["gemm_tiled"] ~~~ PARALLEL["gemm_parallel"] ~~~ SFMAX["softmax"]
    end
    subgraph "Data Layer"
        direction LR
        TENSOR["Tensor"] ~~~ TIMER["Timer"]
    end

    BENCH --> NAIVE & TILED & PARALLEL & SFMAX
    BENCH --> TIMER
    NAIVE & TILED & PARALLEL & SFMAX --> TENSOR

    style BENCH fill:#9C27B0,color:#fff
    style NAIVE fill:#f44336,color:#fff
    style TILED fill:#FF9800,color:#fff
    style PARALLEL fill:#4CAF50,color:#fff
    style SFMAX fill:#2196F3,color:#fff
    style TENSOR fill:#607D8B,color:#fff
    style TIMER fill:#607D8B,color:#fff
```

---

## How It Works

### 1. The Tensor — Your Data Container

Think of a `Tensor` like a spreadsheet grid. It has rows and columns, and each cell holds a number. But in memory, there's no grid — it's a flat list of numbers read left-to-right, top-to-bottom (**row-major** layout):

```mermaid
graph TD
    subgraph "Logical View (3x4 grid)"
        direction LR
        R0["Row 0: 1.0 | 2.0 | 3.0 | 4.0"]
        R1["Row 1: 5.0 | 6.0 | 7.0 | 8.0"]
        R2["Row 2: 9.0 | 10.0 | 11.0 | 12.0"]
    end

    subgraph "Memory Layout (flat array)"
        MEM["[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]"]
    end

    R0 & R1 & R2 --> MEM

    style R0 fill:#4CAF50,color:#fff
    style R1 fill:#FF9800,color:#fff
    style R2 fill:#2196F3,color:#fff
    style MEM fill:#607D8B,color:#fff
```

To find element at row 2, col 1: `index = 2 * 4 + 1 = 9` -> `10.0`

### 2. Matrix Multiplication (GEMM)

GEMM = **G**eneral **M**atrix **M**ultiply. It computes `C = A x B`. For each output cell, take a row from A and a column from B, multiply pair-by-pair, and sum:

```
A (2x3)         B (3x2)         C (2x2)

| 1  2  3 |     | 7   8 |      |  58  64 |
| 4  5  6 |  x  | 9  10 |  =   | 139 154 |
                | 11  12 |

C[0][0] = (1*7) + (2*9) + (3*11) = 58
C[0][1] = (1*8) + (2*10) + (3*12) = 64
```

**Why does this matter?** Every neural network is mostly matrix multiplications — GEMM is the bottleneck.

### 3. The Cache Problem — Why Naive Is Slow

Your CPU has fast but tiny **cache** and slow but large **RAM**. Naive GEMM reads columns of B, which jump around in memory — thrashing the cache:

```mermaid
graph LR
    subgraph "Memory Hierarchy"
        direction LR
        CPU["CPU
        registers"] -->|"~1 ns"| L1["L1 Cache
        ~64 KB"]
        L1 -->|"~3 ns"| L2["L2 Cache
        ~256 KB"]
        L2 -->|"~10 ns"| L3["L3 Cache
        ~8 MB"]
        L3 -->|"~100 ns"| RAM["RAM
        ~16 GB"]
    end

    style CPU fill:#4CAF50,color:#fff
    style L1 fill:#8BC34A,color:#fff
    style L2 fill:#FF9800,color:#fff
    style L3 fill:#FF5722,color:#fff
    style RAM fill:#f44336,color:#fff
```

```mermaid
graph TD
    subgraph "Naive: reading column of B"
        direction LR
        B0["B[0,0] = 7"] -.->|"skip"| B1["B[0,1] = 8"]
        B1 -.->|"skip"| B2["B[1,0] = 9"]
        B2 -.->|"skip"| B3["B[1,1] = 10"]
        B3 -.->|"skip"| B4["B[2,0] = 11"]
        B4 -.->|"skip"| B5["B[2,1] = 12"]
    end

    NEED["Need col 0:
    indices 0, 2, 4
    jumping by stride!"] -->|"cache misses"| B0

    style NEED fill:#f44336,color:#fff
    style B0 fill:#4CAF50,color:#fff
    style B1 fill:#607D8B,color:#fff
    style B2 fill:#4CAF50,color:#fff
    style B3 fill:#607D8B,color:#fff
    style B4 fill:#4CAF50,color:#fff
    style B5 fill:#607D8B,color:#fff
```

For 1024x1024 matrices, B is ~4MB. The cache can't hold it all, so naive GEMM keeps loading and evicting the same data.

### 4. Tiled GEMM — The Fix

Think of it like washing dishes:
- **Naive** = pick up one dish, wash it, put it down, repeat. No strategy.
- **Tiled** = group dishes into stacks. Wash one whole stack before moving on.

Instead of one cell at a time, grab a small **block** of A and B that fits in cache, do all the work, then move on:

```mermaid
graph TD
    subgraph "Naive: cell by cell across whole matrix"
        direction LR
        C1["C[0,0]"] --> C2["C[0,1]"] --> C3["C[0,2]"] --> C4["C[0,3]"] --> C5["..."]
    end

    subgraph "Tiled: block by block, each fits in cache"
        direction LR
        T1["Block 0,0
        16x16"] --> T2["Block 0,1
        16x16"] --> T3["Block 1,0
        16x16"] --> T4["Block 1,1
        16x16"]
    end

    style C1 fill:#f44336,color:#fff
    style C2 fill:#f44336,color:#fff
    style C3 fill:#f44336,color:#fff
    style C4 fill:#f44336,color:#fff
    style C5 fill:#f44336,color:#fff
    style T1 fill:#4CAF50,color:#fff
    style T2 fill:#4CAF50,color:#fff
    style T3 fill:#4CAF50,color:#fff
    style T4 fill:#4CAF50,color:#fff
```

A 16x16 block = 1KB — fits easily in L1 cache. Every access is fast while you're inside that block.

**Same math, same result, just a smarter order of operations.**

### 5. Block Size — Finding the Sweet Spot

```mermaid
graph LR
    BS1["bs=1
    No benefit
    (back to naive)"] -->|"increase"| BS16["bs=16
    Fits in L1
    (sweet spot)"] -->|"increase"| BS32["bs=32
    Fits in L2
    (still good)"] -->|"increase"| BS1024["bs=1024
    Cache overflow
    (back to slow)"]

    style BS1 fill:#f44336,color:#fff
    style BS16 fill:#4CAF50,color:#fff
    style BS32 fill:#8BC34A,color:#fff
    style BS1024 fill:#f44336,color:#fff
```

The benchmark sweeps multiple block sizes to find what works best on your hardware.

### 6. The Benchmark Harness

```mermaid
flowchart LR
    A["Random A, B
    (fixed seed)"] --> B["Warmup
    2 runs"] --> C["Time
    5 trials"] --> D["Average
    time"] --> E["GFLOPS
    2*N^3 / sec / 1e9"]

    style A fill:#607D8B,color:#fff
    style B fill:#FF9800,color:#fff
    style C fill:#4CAF50,color:#fff
    style D fill:#2196F3,color:#fff
    style E fill:#9C27B0,color:#fff
```

Higher GFLOPS = faster kernel. The benchmark also sweeps thread counts (1, 2, 4, 8, ...) to show parallel scaling. Example output:

```
=== N=1024 ===
  naive                      5454.15 ms      0.39 GFLOPS   (baseline)
  tiled  bs=16                646.11 ms      3.32 GFLOPS     8.4x
  tiled  bs=32               1628.45 ms      1.32 GFLOPS     3.3x
  tiled  bs=64               1592.13 ms      1.35 GFLOPS     3.4x
  parallel  bs=16             679.32 ms      3.16 GFLOPS     8.0x
  parallel  bs=16  t=2        351.27 ms      6.11 GFLOPS    15.5x
  parallel  bs=16  t=4        238.11 ms      9.02 GFLOPS    22.9x
  parallel  bs=16  t=8        169.80 ms     12.65 GFLOPS    32.1x
  parallel  bs=16  t=16       128.50 ms     16.71 GFLOPS    42.4x
  parallel  bs=16  t=20       119.22 ms     18.01 GFLOPS    45.8x
  best: parallel bs=16 t=20
```

### 7. Parallel GEMM — Multi-Core Scaling

The parallel kernel splits output tiles across CPU cores using OpenMP. Each core owns distinct tiles, so no synchronization is needed:

```mermaid
graph TD
    subgraph "Thread 0"
        direction LR
        T0A["Tile 0,0"] ~~~ T0B["Tile 0,1"]
    end
    subgraph "Thread 1"
        direction LR
        T1A["Tile 1,0"] ~~~ T1B["Tile 1,1"]
    end
    subgraph "Thread 2"
        direction LR
        T2A["Tile 2,0"] ~~~ T2B["Tile 2,1"]
    end
    subgraph "Thread 3"
        direction LR
        T3A["Tile 3,0"] ~~~ T3B["Tile 3,1"]
    end

    style T0A fill:#4CAF50,color:#fff
    style T0B fill:#4CAF50,color:#fff
    style T1A fill:#FF9800,color:#fff
    style T1B fill:#FF9800,color:#fff
    style T2A fill:#2196F3,color:#fff
    style T2B fill:#2196F3,color:#fff
    style T3A fill:#9C27B0,color:#fff
    style T3B fill:#9C27B0,color:#fff
```

The thread scaling sweep reveals diminishing returns — at small N (128), too many threads actually *hurts* because the overhead of creating/synchronizing threads exceeds the compute time. At large N (1024), more threads keep helping up to the hardware limit.

### 8. Kernel Comparison

| Kernel | Strategy | Cache | Cores | Speedup (N=1024) |
|--------|----------|-------|-------|------------------|
| `gemm_naive` | Triple loop, one cell | Poor | 1 | 1x (baseline) |
| `gemm_tiled` | Block loop, one tile | Good | 1 | ~8x |
| `gemm_parallel` | Block loop, split tiles | Good | All | ~46x |

### 9. Coming Next — Softmax (The Hard Parallelism Problem)

GEMM tiles are **independent** — no communication needed. Softmax requires **reduction** across tiles: each tile computes a partial max and sum, then they must exchange results before normalizing. This mirrors how Graphcore's IPU handles BSP (Bulk Synchronous Parallel) communication.

```mermaid
graph TD
    subgraph "GEMM: tiles work independently"
        direction LR
        G1["Tile A
        no comms"] ~~~ G2["Tile B
        no comms"] ~~~ G3["Tile C
        no comms"] ~~~ G4["Tile D
        no comms"]
    end

    subgraph "Softmax: tiles must reduce together"
        S1["Tile A
        partial max/sum"] --> R["Reduce
        global max + sum"]
        S2["Tile B
        partial max/sum"] --> R
        S3["Tile C
        partial max/sum"] --> R
        S4["Tile D
        partial max/sum"] --> R
        R --> F["Normalize
        exp(x-max) / sum"]
    end

    style G1 fill:#4CAF50,color:#fff
    style G2 fill:#4CAF50,color:#fff
    style G3 fill:#4CAF50,color:#fff
    style G4 fill:#4CAF50,color:#fff
    style S1 fill:#FF9800,color:#fff
    style S2 fill:#FF9800,color:#fff
    style S3 fill:#FF9800,color:#fff
    style S4 fill:#FF9800,color:#fff
    style R fill:#f44336,color:#fff
    style F fill:#9C27B0,color:#fff
```

---

## Current Status

Phases 1-6 are complete:

- **Tensor class** — row-major `std::vector<float>` storage with bounds-checked `at()`, raw `data()` pointer, `fill`, `zero`, `randomize(seed)`
- **Naive GEMM** — triple-loop `gemm_naive(A, B, C)` with dimension validation
- **Tiled GEMM** — cache-friendly block-based `gemm_tiled(A, B, C, block_size)` with configurable tile size
- **Parallel GEMM** — OpenMP-parallelized tiled GEMM with `collapse(2)` over output tiles, thread count configurable at runtime
- **Timer** — `std::chrono`-based high-resolution timer for benchmarking
- **Benchmark harness** — grouped output by matrix size, GFLOPS reporting, speedup vs baseline, thread scaling sweep (1/2/4/8/.../max), best-kernel summary
- **Test suite** — tensor tests (11 cases) and GEMM correctness tests (15 cases: 7 naive + 4 tiled + 4 parallel) using a minimal in-repo assertion framework

Upcoming: softmax kernels, scheduler abstraction, docs.

## Repository Layout

```text
.
├── benchmarks/
│   └── benchmark_gemm.cpp        # benchmark harness (GFLOPS, speedup, best-kernel)
├── docs/
│   └── design_decisions.md       # upfront design decisions
├── include/
│   ├── tensor.h                  # Tensor class
│   ├── gemm.h                    # GEMM kernel declarations
│   └── timer.h                   # Timer class
├── src/
│   ├── tensor.cpp                # Tensor implementation
│   ├── gemm_naive.cpp            # naive GEMM kernel
│   ├── gemm_tiled.cpp            # tiled GEMM kernel
│   └── gemm_parallel.cpp         # OpenMP parallel GEMM kernel
├── tests/
│   ├── test_utils.h              # assertion macros
│   ├── test_tensor.cpp           # tensor tests (11 cases)
│   └── test_gemm.cpp             # GEMM correctness tests (15 cases)
├── CMakeLists.txt
├── Makefile
└── tile_runtime.plan.md
```

## Quick Start

```bash
make          # build everything
make test     # run all tests
make bench    # run benchmarks
make clean    # remove build directory
```

## Notes

- OpenMP is detected and linked automatically when available.
- All code lives in the `tile_runtime::` namespace.
- See [docs/design_decisions.md](docs/design_decisions.md) for design rationale.
