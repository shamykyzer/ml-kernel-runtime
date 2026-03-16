#include "gemm.h"

#include <stdexcept>

namespace tile_runtime {

void gemm_naive(const Tensor& A, const Tensor& B, Tensor& C) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument(
            "gemm_naive: A.cols() != B.rows() — dimension mismatch");
    }
    if (C.rows() != A.rows() || C.cols() != B.cols()) {
        throw std::invalid_argument(
            "gemm_naive: C dimensions do not match (A.rows() x B.cols())");
    }

    const size_t M = A.rows();
    const size_t K = A.cols();
    const size_t N = B.cols();

    const float* a = A.data();
    const float* b = B.data();
    float* c = C.data();

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

}  // namespace tile_runtime
