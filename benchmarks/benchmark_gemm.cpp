#include "tensor.h"
#include "gemm.h"
#include "timer.h"

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using tile_runtime::Tensor;
using tile_runtime::Timer;

struct BenchResult {
    std::string label;
    size_t block_size;
    int threads;
    double time_ms;
    double gflops;
};

static double compute_gflops(size_t N, double seconds) {
    return 2.0 * N * N * N / seconds / 1e9;
}

using KernelFn = void(*)(const Tensor&, const Tensor&, Tensor&);
using TiledKernelFn = void(*)(const Tensor&, const Tensor&, Tensor&, size_t);

static int get_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static void set_num_threads(int t) {
#ifdef _OPENMP
    omp_set_num_threads(t);
#else
    (void)t;
#endif
}

static BenchResult bench_kernel(const std::string& label, size_t N,
                                KernelFn fn, int warmup, int trials) {
    Tensor A(N, N), B(N, N), C(N, N);
    A.randomize(42);
    B.randomize(84);

    for (int w = 0; w < warmup; ++w) {
        C.zero();
        fn(A, B, C);
    }

    Timer timer;
    double total_ms = 0.0;
    for (int t = 0; t < trials; ++t) {
        C.zero();
        timer.start();
        fn(A, B, C);
        timer.stop();
        total_ms += timer.elapsed_ms();
    }

    double avg_ms = total_ms / trials;
    double avg_sec = avg_ms / 1000.0;
    return {label, 0, 1, avg_ms, compute_gflops(N, avg_sec)};
}

static BenchResult bench_tiled(const std::string& label, size_t N,
                               size_t block_size, TiledKernelFn fn,
                               int warmup, int trials) {
    Tensor A(N, N), B(N, N), C(N, N);
    A.randomize(42);
    B.randomize(84);

    for (int w = 0; w < warmup; ++w) {
        C.zero();
        fn(A, B, C, block_size);
    }

    Timer timer;
    double total_ms = 0.0;
    for (int t = 0; t < trials; ++t) {
        C.zero();
        timer.start();
        fn(A, B, C, block_size);
        timer.stop();
        total_ms += timer.elapsed_ms();
    }

    double avg_ms = total_ms / trials;
    double avg_sec = avg_ms / 1000.0;
    return {label, block_size, 1, avg_ms, compute_gflops(N, avg_sec)};
}

static BenchResult bench_parallel(const std::string& label, size_t N,
                                  size_t block_size, TiledKernelFn fn,
                                  int warmup, int trials, int threads) {
    Tensor A(N, N), B(N, N), C(N, N);
    A.randomize(42);
    B.randomize(84);

    set_num_threads(threads);
    int actual_threads = get_max_threads();

    for (int w = 0; w < warmup; ++w) {
        C.zero();
        fn(A, B, C, block_size);
    }

    Timer timer;
    double total_ms = 0.0;
    for (int t = 0; t < trials; ++t) {
        C.zero();
        timer.start();
        fn(A, B, C, block_size);
        timer.stop();
        total_ms += timer.elapsed_ms();
    }

    double avg_ms = total_ms / trials;
    double avg_sec = avg_ms / 1000.0;
    return {label, block_size, actual_threads, avg_ms, compute_gflops(N, avg_sec)};
}

static void print_baseline(const BenchResult& r) {
    std::cout << "  " << std::left << std::setw(24) << r.label
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(10) << r.time_ms << " ms"
              << std::setw(10) << r.gflops << " GFLOPS"
              << "   (baseline)" << std::endl;
}

static void print_row(const BenchResult& r, double baseline_ms) {
    double speedup = (r.time_ms > 0.0) ? baseline_ms / r.time_ms : 0.0;
    std::string label = r.label;
    if (r.block_size > 0) {
        label += "  bs=" + std::to_string(r.block_size);
    }
    if (r.threads > 1) {
        label += "  t=" + std::to_string(r.threads);
    }
    std::cout << "  " << std::left << std::setw(24) << label
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(10) << r.time_ms << " ms"
              << std::setw(10) << r.gflops << " GFLOPS"
              << std::setw(8) << std::setprecision(1) << speedup << "x"
              << std::endl;
}

// Build thread counts to sweep: 1, 2, 4, ... up to max
static std::vector<int> thread_counts(int max_t) {
    std::vector<int> counts;
    for (int t = 1; t <= max_t; t *= 2) {
        counts.push_back(t);
    }
    if (counts.back() != max_t) {
        counts.push_back(max_t);
    }
    return counts;
}

int main() {
    const std::vector<size_t> sizes = {128, 256, 512, 1024};
    const std::vector<size_t> block_sizes = {16, 32, 64};
    const int warmup = 2;
    const int trials = 5;
    const int max_threads = get_max_threads();

    std::cout << "=== ML Kernel Runtime Benchmark ===" << std::endl;
    std::cout << "Warmup: " << warmup << "  Trials: " << trials
              << "  Max threads: " << max_threads << std::endl;

    for (size_t N : sizes) {
        std::cout << std::endl;
        std::cout << "=== N=" << N << " ===" << std::endl;

        // --- Naive baseline ---
        auto naive = bench_kernel("naive", N, tile_runtime::gemm_naive,
                                  warmup, trials);
        print_baseline(naive);

        // --- Tiled variants (single-threaded) ---
        std::string best_label;
        double best_ms = naive.time_ms;
        size_t best_bs = 16;

        for (size_t bs : block_sizes) {
            auto tiled = bench_tiled("tiled", N, bs,
                                     tile_runtime::gemm_tiled,
                                     warmup, trials);
            print_row(tiled, naive.time_ms);

            if (tiled.time_ms < best_ms) {
                best_ms = tiled.time_ms;
                best_bs = bs;
                best_label = "tiled bs=" + std::to_string(bs);
            }
        }

        // --- Parallel thread scaling (use best block size) ---
        auto threads = thread_counts(max_threads);
        for (int t : threads) {
            auto par = bench_parallel("parallel", N, best_bs,
                                      tile_runtime::gemm_parallel,
                                      warmup, trials, t);
            print_row(par, naive.time_ms);

            if (par.time_ms < best_ms) {
                best_ms = par.time_ms;
                best_label = "parallel bs=" + std::to_string(best_bs)
                           + " t=" + std::to_string(par.threads);
            }
        }

        // Restore max threads for next iteration
        set_num_threads(max_threads);

        if (!best_label.empty()) {
            std::cout << "  best: " << best_label << std::endl;
        } else {
            std::cout << "  best: naive" << std::endl;
        }
    }

    std::cout << std::endl;
    return 0;
}
