#pragma once

#include "tensor.h"
#include <cstddef>

namespace tile_runtime {

// Naive triple-loop GEMM: C = A * B
// Requires: A.cols() == B.rows(), C sized to (A.rows() x B.cols()).
void gemm_naive(const Tensor& A, const Tensor& B, Tensor& C);

// Cache-friendly tiled GEMM: C = A * B
// block_size controls the tile dimension for ii/jj/kk loops.
void gemm_tiled(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size);

// OpenMP-parallelized tiled GEMM: C = A * B
// Parallelizes over independent output tiles.
void gemm_parallel(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size);

}  // namespace tile_runtime
