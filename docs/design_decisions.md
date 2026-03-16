# Design Decisions

Resolved upfront to keep implementation consistent across all phases.

## 1. Bounds Checking

`Tensor::at(i, j)` always validates indices and throws `std::out_of_range` on
violation.  For kernel inner loops where overhead matters, use the raw pointer
returned by `Tensor::data()` with manual `row * cols + col` indexing.

## 2. Output Tensor Ownership

All GEMM functions require the caller to pre-allocate `C` with correct
dimensions.  Kernels validate dimensions at entry and throw `std::invalid_argument`
on mismatch.  No implicit resizing — this keeps the API predictable and avoids
hidden allocations in hot paths.

## 3. Memory Layout

Row-major, contiguous `std::vector<float>`.  The design does not preclude
switching to `aligned_alloc` storage later for cache-line alignment (64-byte)
or SIMD-friendly padding.

## 4. Namespace

Everything lives in `tile_runtime::`.  Kernel functions are free functions in
that namespace, not methods on `Tensor`.

## 5. Float Comparison Tolerance

Correctness tests use a relative tolerance of `1e-4f`, defined in
`tests/test_utils.h`.  Relative error is used for values away from zero;
absolute tolerance is used for values near zero.

## 6. Test Framework

Minimal in-repo assertion macros in `tests/test_utils.h`:
`ASSERT_TRUE`, `ASSERT_NEAR`, `ASSERT_THROWS`.  Each test file has its own
`main()` and returns nonzero on failure.  CMake registers each as a CTest.
