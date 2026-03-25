#include "tensor.h"
#include "gemm.h"
#include "timer.h"
#include "cpu_features.h"

#include <cstddef>
#include <fstream>
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
    size_t N;
    size_t block_size;
    int threads;
    double time_ms;
    double gflops;
    double speedup;
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
    return {label, N, 0, 1, avg_ms, compute_gflops(N, avg_sec), 0.0};
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
    return {label, N, block_size, 1, avg_ms, compute_gflops(N, avg_sec), 0.0};
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
    return {label, N, block_size, actual_threads, avg_ms, compute_gflops(N, avg_sec), 0.0};
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
    for (int t = 1; t <= max_t; t += (t < 4 ? 1 : 2)) {
        counts.push_back(t);
    }
    if (counts.back() != max_t) {
        counts.push_back(max_t);
    }
    return counts;
}

int main() {
    const std::vector<size_t> sizes = {128, 256, 512, 1024};
    const std::vector<size_t> block_sizes = {16, 32, 48, 64, 128};
    const int warmup = 2;
    const int trials = 5;
    const int max_threads = get_max_threads();

    const auto& cpu = tile_runtime::CpuFeatures::detect();

    std::cout << "=== ML Kernel Runtime Benchmark ===" << std::endl;
    std::cout << "Warmup: " << warmup << "  Trials: " << trials
              << "  Max threads: " << max_threads << std::endl;
    std::cout << "CPU: AVX2=" << cpu.avx2 << " FMA=" << cpu.fma
              << " AVX512F=" << cpu.avx512f << " AVX512VL=" << cpu.avx512vl
              << std::endl;

    std::vector<BenchResult> all_results;

    for (size_t N : sizes) {
        std::cout << std::endl;
        std::cout << "=== N=" << N << " ===" << std::endl;

        // --- Naive baseline ---
        auto naive = bench_kernel("naive", N, tile_runtime::gemm_naive,
                                  warmup, trials);
        naive.speedup = 1.0;
        print_baseline(naive);
        all_results.push_back(naive);

        // --- Tiled variants (single-threaded) ---
        std::string best_label;
        double best_ms = naive.time_ms;
        size_t best_bs = 16;

        auto record = [&](BenchResult& r) {
            r.speedup = (r.time_ms > 0.0) ? naive.time_ms / r.time_ms : 0.0;
            all_results.push_back(r);
            if (r.time_ms < best_ms) {
                best_ms = r.time_ms;
                best_bs = r.block_size > 0 ? r.block_size : best_bs;
                best_label = r.label;
                if (r.block_size > 0) best_label += " bs=" + std::to_string(r.block_size);
                if (r.threads > 1) best_label += " t=" + std::to_string(r.threads);
            }
        };

        for (size_t bs : block_sizes) {
            auto tiled = bench_tiled("tiled", N, bs,
                                     tile_runtime::gemm_tiled,
                                     warmup, trials);
            print_row(tiled, naive.time_ms);
            record(tiled);
        }

        // --- Parallel thread scaling (sweep block sizes) ---
        auto threads = thread_counts(max_threads);
        for (size_t bs : block_sizes) {
            for (int t : threads) {
                auto par = bench_parallel("parallel", N, bs,
                                          tile_runtime::gemm_parallel,
                                          warmup, trials, t);
                print_row(par, naive.time_ms);
                record(par);
            }
        }

        // --- SIMD kernels (sweep block sizes) ---
        if (cpu.avx2 && cpu.fma) {
            for (size_t bs : block_sizes) {
                auto avx = bench_tiled("avx2+fma", N, bs,
                                        tile_runtime::gemm_avx, warmup, trials);
                print_row(avx, naive.time_ms);
                record(avx);
            }
        }

        if (cpu.avx512f) {
            for (size_t bs : block_sizes) {
                auto avx512 = bench_tiled("avx-512", N, bs,
                                           tile_runtime::gemm_avx512, warmup, trials);
                print_row(avx512, naive.time_ms);
                record(avx512);
            }
        }

#ifdef TILE_HAS_STD_SIMD
        for (size_t bs : block_sizes) {
            auto simd = bench_tiled("std-simd", N, bs,
                                     tile_runtime::gemm_simd, warmup, trials);
            print_row(simd, naive.time_ms);
            record(simd);
        }
#endif

        // --- Parallel + SIMD (AVX2, sweep block sizes) ---
        if (cpu.avx2 && cpu.fma) {
            for (size_t bs : block_sizes) {
                for (int t : threads) {
                    auto par_simd = bench_parallel("parallel+simd", N, bs,
                                                    tile_runtime::gemm_parallel_simd,
                                                    warmup, trials, t);
                    print_row(par_simd, naive.time_ms);
                    record(par_simd);
                }
            }
        }

        // --- Parallel + AVX-512 (sweep block sizes) ---
        if (cpu.avx512f) {
            for (size_t bs : block_sizes) {
                for (int t : threads) {
                    auto par_avx512 = bench_parallel("parallel+avx512", N, bs,
                                                      tile_runtime::gemm_parallel_avx512,
                                                      warmup, trials, t);
                    print_row(par_avx512, naive.time_ms);
                    record(par_avx512);
                }
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

    // --- Write CSV ---
    std::ofstream csv("benchmark_results.csv");
    if (csv.is_open()) {
        csv << "kernel,N,block_size,threads,time_ms,gflops,speedup\n";
        csv << std::fixed;
        for (const auto& r : all_results) {
            csv << r.label << ","
                << r.N << ","
                << r.block_size << ","
                << r.threads << ","
                << std::setprecision(2) << r.time_ms << ","
                << std::setprecision(2) << r.gflops << ","
                << std::setprecision(1) << r.speedup << "\n";
        }
        csv.close();
        std::cout << "Results written to benchmark_results.csv" << std::endl;
    }

    return 0;
}
