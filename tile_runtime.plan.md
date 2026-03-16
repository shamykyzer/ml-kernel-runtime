---
name: tile runtime build
overview: "Turn the current blank repo into a small, benchmarked C++17 tile-based GEMM runtime with a staged build-out: tensor foundation, naive/tiled/parallel kernels, scheduler abstraction, tests, benchmarks, and docs."
todos:
  - id: bootstrap-layout
    content: Create the repo structure and CMake build with a core library, benchmark executable, and test executable.
    status: done
  - id: tensor-foundation
    content: Implement the Tensor type with row-major storage, utility methods, and tensor-focused tests.
    status: done
  - id: kernel-baseline
    content: Implement naive GEMM and use it as the correctness reference for all later kernels.
    status: done
  - id: benchmark-harness
    content: Add timer utilities and a benchmark runner that reports time, GFLOPS, block size, and thread count.
    status: done
  - id: kernel-optimizations
    content: Implement tiled GEMM and OpenMP parallel GEMM, then validate correctness and measure speedups.
    status: in-progress
  - id: runtime-layer
    content: Introduce TileTask generation and a minimal Scheduler that dispatches tile-local GEMM work.
    status: pending
  - id: docs-polish
    content: Write README and docs covering architecture, performance analysis, and Graphcore-inspired runtime design.
    status: pending
isProject: false
---

# Tile Runtime Implementation Plan

## Current State

- Existing repo content is minimal: [README.md](/home/shamy/ml-kernel-runtime/README.md) contains only the project one-line summary.
- No build system, source tree, tests, benchmarks, or docs exist yet, so this should be treated as a greenfield implementation.

## Defaults To Use

- Language/build: C++17 + CMake.
- Parallelism: OpenMP when available.
- Testing: a small in-repo test executable under [tests/](/home/shamy/ml-kernel-runtime/tests) rather than introducing GoogleTest in v1.
- Matrix type: `Tensor` with contiguous row-major `std::vector<float>` storage.
- Benchmark focus: square GEMM first, with repeatable warmup/timing and GFLOPS reporting.

## Phase 1: Bootstrap The Repo

- Create the source layout from your spec: [include/](/home/shamy/ml-kernel-runtime/include), [src/](/home/shamy/ml-kernel-runtime/src), [tests/](/home/shamy/ml-kernel-runtime/tests), [benchmarks/](/home/shamy/ml-kernel-runtime/benchmarks), [docs/](/home/shamy/ml-kernel-runtime/docs), and [scripts/](/home/shamy/ml-kernel-runtime/scripts).
- Add [CMakeLists.txt](/home/shamy/ml-kernel-runtime/CMakeLists.txt) that:
  - sets `CMAKE_CXX_STANDARD 17`
  - enables warnings and optimized `Release` flags
  - discovers/links OpenMP if present
  - builds a reusable library target for core runtime code
  - builds separate `benchmark_gemm` and `test_runtime` executables
- Expand [README.md](/home/shamy/ml-kernel-runtime/README.md) into a real project overview with build/run instructions and project goals.

## Phase 2: Build The Core Data Layer

- Implement [include/tensor.h](/home/shamy/ml-kernel-runtime/include/tensor.h) and [src/tensor.cpp](/home/shamy/ml-kernel-runtime/src/tensor.cpp).
- Keep the API close to your spec:

```cpp
  class Tensor {
  public:
      Tensor();
      Tensor(size_t rows, size_t cols);
      Tensor(size_t rows, size_t cols, float init_value);
      float& at(size_t i, size_t j);
      const float& at(size_t i, size_t j) const;
  };
  

```

- Include shape accessors, `fill`, `zero`, `randomize`, and optional raw data accessors for later kernel work.
- Decide once whether out-of-bounds checks live in `at()` always or only in debug builds; keep it consistent.

## Phase 3: Establish Correctness With Naive GEMM

- Add [include/gemm.h](/home/shamy/ml-kernel-runtime/include/gemm.h) and [src/gemm_naive.cpp](/home/shamy/ml-kernel-runtime/src/gemm_naive.cpp).
- Define a stable kernel surface early:

```cpp
  void gemm_naive(const Tensor& A, const Tensor& B, Tensor& C);
  void gemm_tiled(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size);
  void gemm_parallel(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size);
  

```

- Validate dimensions up front and make `C` sizing behavior explicit: either require pre-sized output or resize internally. Pick one rule and document it in the header.
- Use the naive kernel as the golden reference for all later tests.

## Phase 4: Add Timer + Benchmark Harness

- Implement [include/timer.h](/home/shamy/ml-kernel-runtime/include/timer.h) and [src/timer.cpp](/home/shamy/ml-kernel-runtime/src/timer.cpp) with a simple `std::chrono` wrapper.
- Build [benchmarks/benchmark_gemm.cpp](/home/shamy/ml-kernel-runtime/benchmarks/benchmark_gemm.cpp) to:
  - generate deterministic random inputs
  - warm up each kernel once
  - run timed trials for sizes like `128, 256, 512, 1024`
  - compute `GFLOPS = 2 * n^3 / seconds / 1e9`
  - print kernel name, size, block size, thread count, time, GFLOPS, and speedup
- Keep benchmark logic separate from kernels so runtime code stays reusable.

## Phase 5: Optimize With Tiled GEMM

- Implement [src/gemm_tiled.cpp](/home/shamy/ml-kernel-runtime/src/gemm_tiled.cpp) with outer tile loops over `ii`, `jj`, and `kk`.
- Handle non-divisible edges with `std::min` bounds.
- Be explicit about accumulation semantics so partial sums across `kk` tiles are preserved correctly.
- Compare against `gemm_naive` for small and medium random matrices before trusting performance numbers.

## Phase 6: Add Parallel Tile Execution

- Implement [src/gemm_parallel.cpp](/home/shamy/ml-kernel-runtime/src/gemm_parallel.cpp) by parallelizing independent output tiles, not inner reduction work.
- Prefer `#pragma omp parallel for collapse(2)` over the output tile grid (`ii`, `jj`) so each thread owns distinct output regions.
- Make thread count configurable through the benchmark harness rather than hard-coding it in the kernel.
- Test with `OMP_NUM_THREADS=1` first, then scale up.

## Phase 7: Introduce Runtime-Style Tile Tasks

- Add [include/tile.h](/home/shamy/ml-kernel-runtime/include/tile.h) and [src/tile.cpp](/home/shamy/ml-kernel-runtime/src/tile.cpp).
- Define `TileTask` for output-space partitioning with row/col bounds and optional reduction bounds if needed later.
- Implement a helper like `make_tasks(rows, cols, block_size)` that fully covers the output matrix and safely handles edge tiles.
- Keep this layer independent from OpenMP so the task abstraction stays conceptually separate from the execution backend.

## Phase 8: Add A Minimal Scheduler Layer

- Add [include/scheduler.h](/home/shamy/ml-kernel-runtime/include/scheduler.h) and [src/scheduler.cpp](/home/shamy/ml-kernel-runtime/src/scheduler.cpp).
- Start with a deliberately simple scheduler:
  - create the full task list
  - iterate or parallel-iterate across tasks
  - dispatch tile-local GEMM work for each task
- Expose a clean entry point like `Scheduler::run_gemm(...)` so the code reads like a miniature runtime rather than a pile of loops.

```mermaid
flowchart TD
    TensorA[Tensor A] --> Scheduler
    TensorB[Tensor B] --> Scheduler
    Scheduler --> TaskGen[TileTask generation]
    TaskGen --> WorkerLoop[Tile execution loop]
    WorkerLoop --> Kernel[Tile GEMM kernel]
    Kernel --> TensorC[Output Tensor C]
    TensorC --> Benchmarks[Benchmarks and tests]
```



## Phase 9: Build Confidence With Tests

- Add [tests/test_tensor.cpp](/home/shamy/ml-kernel-runtime/tests/test_tensor.cpp) and [tests/test_gemm.cpp](/home/shamy/ml-kernel-runtime/tests/test_gemm.cpp).
- Add [tests/test_scheduler.cpp](/home/shamy/ml-kernel-runtime/tests/test_scheduler.cpp) if scheduler logic becomes non-trivial.
- Cover:
  - tensor shape/indexing/fill/zero/randomize sanity
  - known small GEMM example
  - random naive vs tiled comparisons
  - random tiled vs parallel comparisons
  - tile coverage and edge handling
- Use tolerance-based float comparison helpers rather than exact equality for computed outputs.

## Phase 10: Documentation And Analysis

- Rewrite [README.md](/home/shamy/ml-kernel-runtime/README.md) to present the repo as a systems/performance project.
- Add [docs/architecture.md](/home/shamy/ml-kernel-runtime/docs/architecture.md) explaining the data/kernel/runtime/benchmark layers.
- Add [docs/performance_analysis.md](/home/shamy/ml-kernel-runtime/docs/performance_analysis.md) to summarize baseline, tiling gains, block-size sweep, and thread scaling.
- Add [docs/poplibs_analysis.md](/home/shamy/ml-kernel-runtime/docs/poplibs_analysis.md) to explain the Graphcore-inspired ideas without overstating hardware fidelity.
- Add [scripts/run_benchmarks.sh](/home/shamy/ml-kernel-runtime/scripts/run_benchmarks.sh) for a repeatable benchmark workflow.

## Suggested Delivery Order

1. Bootstrap repo + CMake.
2. Implement `Tensor` + tensor tests.
3. Implement naive GEMM + correctness tests.
4. Add timer + benchmark executable and record baseline.
5. Implement tiled GEMM + compare block sizes.
6. Implement OpenMP parallel GEMM + thread scaling.
7. Add `TileTask` + `Scheduler` abstraction.
8. Finish docs and benchmark analysis.

## Risks To Watch

- Silent indexing bugs in row-major addressing.
- Ambiguous `Tensor C` ownership/resizing semantics across kernels.
- Race conditions if parallel work ever overlaps on the same output tile.
- Misleading benchmark results from debug builds, tiny matrices, or lack of warmup.
- Overcomplicating the scheduler before the kernels are correct and measurable.

## Definition Of Done

- `cmake` configures cleanly and builds in `Release` mode.
- Tensor, naive, tiled, and parallel GEMM all pass correctness tests.
- Benchmark executable reports stable performance trends and GFLOPS.
- Scheduler/task layer works and is reflected in the architecture docs.
- Repo reads as a polished systems project rather than an unfinished coding exercise.

