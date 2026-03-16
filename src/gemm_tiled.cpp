#include "gemm.h"

#include <algorithm>
#include <stdexcept>

namespace tile_runtime {

void gemm_tiled(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument(
            "gemm_tiled: A.cols() != B.rows() — dimension mismatch");
    }
    if (C.rows() != A.rows() || C.cols() != B.cols()) {
        throw std::invalid_argument(
            "gemm_tiled: C dimensions do not match (A.rows() x B.cols())");
    }

    const size_t M = A.rows();
    const size_t K = A.cols();
    const size_t N = B.cols();

    const float* a = A.data();
    const float* b = B.data();
    float* c = C.data();

    for (size_t ii = 0; ii < M; ii += block_size) {
        const size_t i_end = std::min(ii + block_size, M);
        for (size_t jj = 0; jj < N; jj += block_size) {
            const size_t j_end = std::min(jj + block_size, N);
            for (size_t kk = 0; kk < K; kk += block_size) {
                const size_t k_end = std::min(kk + block_size, K);

                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t j = jj; j < j_end; ++j) {
                        float sum = 0.0f;
                        for (size_t k = kk; k < k_end; ++k) {
                            sum += a[i * K + k] * b[k * N + j];
                        }
                        c[i * N + j] += sum;  // accumulate across kk tiles
                    }
                }
            }
        }
    }
}

}  // namespace tile_runtime
