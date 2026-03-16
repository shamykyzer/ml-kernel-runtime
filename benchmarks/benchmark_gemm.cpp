#include "tensor.h"
#include "gemm.h"
#include "timer.h"

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

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

static BenchResult bench_tiled_kernel(const std::string& label, size_t N,
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

static void print_baseline(const BenchResult& r) {
    std::cout << "  " << std::left << std::setw(18) << r.label
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(10) << r.time_ms << " ms"
              << std::setw(10) << r.gflops << " GFLOPS"
              << "   (baseline)" << std::endl;
}

static void print_row(const BenchResult& r, double baseline_ms) {
    double speedup = (r.time_ms > 0.0) ? baseline_ms / r.time_ms : 0.0;
    std::string label = r.label + "  bs=" + std::to_string(r.block_size);
    std::cout << "  " << std::left << std::setw(18) << label
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(10) << r.time_ms << " ms"
              << std::setw(10) << r.gflops << " GFLOPS"
              << std::setw(8) << std::setprecision(1) << speedup << "x"
              << std::endl;
}

int main() {
    const std::vector<size_t> sizes = {128, 256, 512, 1024};
    const std::vector<size_t> block_sizes = {16, 32, 64};
    const int warmup = 2;
    const int trials = 5;

    std::cout << "=== ML Kernel Runtime Benchmark ===" << std::endl;
    std::cout << "Warmup: " << warmup << "  Trials: " << trials << std::endl;

    for (size_t N : sizes) {
        std::cout << std::endl;
        std::cout << "=== N=" << N << " ===" << std::endl;

        // Naive baseline
        auto naive = bench_kernel("naive", N, tile_runtime::gemm_naive,
                                  warmup, trials);
        print_baseline(naive);

        // Tiled variants — track the best
        std::string best_label;
        double best_ms = naive.time_ms;

        for (size_t bs : block_sizes) {
            auto tiled = bench_tiled_kernel("tiled", N, bs,
                                            tile_runtime::gemm_tiled,
                                            warmup, trials);
            print_row(tiled, naive.time_ms);

            if (tiled.time_ms < best_ms) {
                best_ms = tiled.time_ms;
                best_label = "tiled bs=" + std::to_string(bs);
            }
        }

        if (!best_label.empty()) {
            std::cout << "  best: " << best_label << std::endl;
        } else {
            std::cout << "  best: naive" << std::endl;
        }
    }

    std::cout << std::endl;
    return 0;
}
