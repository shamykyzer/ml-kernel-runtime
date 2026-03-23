#pragma once

namespace tile_runtime {

struct CpuFeatures {
    bool avx2;
    bool fma;
    bool avx512f;
    bool avx512vl;

    static const CpuFeatures& detect() {
        static const CpuFeatures features = query();
        return features;
    }

private:
    static CpuFeatures query() {
        CpuFeatures f{};
#if defined(__GNUC__) || defined(__clang__)
        __builtin_cpu_init();
        f.avx2     = __builtin_cpu_supports("avx2");
        f.fma      = __builtin_cpu_supports("fma");
        f.avx512f  = __builtin_cpu_supports("avx512f");
        f.avx512vl = __builtin_cpu_supports("avx512vl");
#elif defined(_MSC_VER)
        int info[4];
        __cpuidex(info, 1, 0);
        f.fma = (info[2] >> 12) & 1;
        __cpuidex(info, 7, 0);
        f.avx2     = (info[1] >> 5) & 1;
        f.avx512f  = (info[1] >> 16) & 1;
        f.avx512vl = (info[1] >> 31) & 1;
#endif
        return f;
    }
};

}  // namespace tile_runtime
