#include "test_utils.h"
#include "gemm.h"
#include "tensor.h"

using tile_runtime::Tensor;
using tile_runtime::gemm_naive;
using tile_runtime::gemm_tiled;

// Helper: manual dot-product reference (not calling gemm_naive).
static void reference_matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < B.cols(); ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < A.cols(); ++k) {
                sum += A.data()[i * A.cols() + k] * B.data()[k * B.cols() + j];
            }
            C.data()[i * C.cols() + j] = sum;
        }
    }
}

void test_2x2_known() {
    // | 1 2 |   | 5 6 |   | 19 22 |
    // | 3 4 | x | 7 8 | = | 43 50 |
    Tensor A(2, 2), B(2, 2), C(2, 2);
    A.at(0,0)=1; A.at(0,1)=2; A.at(1,0)=3; A.at(1,1)=4;
    B.at(0,0)=5; B.at(0,1)=6; B.at(1,0)=7; B.at(1,1)=8;

    gemm_naive(A, B, C);

    ASSERT_NEAR(C.at(0,0), 19.0f);
    ASSERT_NEAR(C.at(0,1), 22.0f);
    ASSERT_NEAR(C.at(1,0), 43.0f);
    ASSERT_NEAR(C.at(1,1), 50.0f);
}

void test_rectangular_3x2_times_2x4() {
    // A (3x2), B (2x4) -> C (3x4)
    Tensor A(3, 2), B(2, 4), C(3, 4);
    A.at(0,0)=1; A.at(0,1)=2;
    A.at(1,0)=3; A.at(1,1)=4;
    A.at(2,0)=5; A.at(2,1)=6;

    B.at(0,0)=1; B.at(0,1)=2; B.at(0,2)=3; B.at(0,3)=4;
    B.at(1,0)=5; B.at(1,1)=6; B.at(1,2)=7; B.at(1,3)=8;

    gemm_naive(A, B, C);

    // Row 0: (1*1+2*5, 1*2+2*6, 1*3+2*7, 1*4+2*8) = (11,14,17,20)
    ASSERT_NEAR(C.at(0,0), 11.0f);
    ASSERT_NEAR(C.at(0,1), 14.0f);
    ASSERT_NEAR(C.at(0,2), 17.0f);
    ASSERT_NEAR(C.at(0,3), 20.0f);
    // Row 1: (3*1+4*5, 3*2+4*6, 3*3+4*7, 3*4+4*8) = (23,30,37,44)
    ASSERT_NEAR(C.at(1,0), 23.0f);
    ASSERT_NEAR(C.at(1,1), 30.0f);
    ASSERT_NEAR(C.at(1,2), 37.0f);
    ASSERT_NEAR(C.at(1,3), 44.0f);
    // Row 2: (5*1+6*5, 5*2+6*6, 5*3+6*7, 5*4+6*8) = (35,46,57,68)
    ASSERT_NEAR(C.at(2,0), 35.0f);
    ASSERT_NEAR(C.at(2,1), 46.0f);
    ASSERT_NEAR(C.at(2,2), 57.0f);
    ASSERT_NEAR(C.at(2,3), 68.0f);
}

void test_identity_multiplication() {
    Tensor A(3, 3);
    A.randomize(42);

    // Build identity
    Tensor I(3, 3);
    for (size_t i = 0; i < 3; ++i)
        I.at(i, i) = 1.0f;

    Tensor C(3, 3);
    gemm_naive(A, I, C);

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            ASSERT_NEAR(C.at(i,j), A.at(i,j));
}

void test_zero_matrix() {
    Tensor A(3, 3);
    A.randomize(10);

    Tensor Z(3, 3);  // all zeros

    Tensor C(3, 3, 999.0f);
    gemm_naive(A, Z, C);

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            ASSERT_NEAR(C.at(i,j), 0.0f);
}

void test_dimension_mismatch() {
    Tensor A(2, 3), B(4, 2), C(2, 2);
    ASSERT_THROWS(gemm_naive(A, B, C), std::invalid_argument);

    // Correct inner dims but wrong C size
    Tensor A2(2, 3), B2(3, 4), C2(2, 2);
    ASSERT_THROWS(gemm_naive(A2, B2, C2), std::invalid_argument);
}

void test_random_vs_reference() {
    size_t sizes[] = {1, 7, 16, 64};
    for (size_t n : sizes) {
        Tensor A(n, n), B(n, n), C(n, n), Ref(n, n);
        A.randomize(100 + static_cast<unsigned>(n));
        B.randomize(200 + static_cast<unsigned>(n));

        gemm_naive(A, B, C);
        reference_matmul(A, B, Ref);

        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                ASSERT_NEAR(C.at(i,j), Ref.at(i,j));
    }
}

void test_1x1() {
    Tensor A(1, 1, 3.0f), B(1, 1, 7.0f), C(1, 1);
    gemm_naive(A, B, C);
    ASSERT_NEAR(C.at(0,0), 21.0f);
}

// --- Tiled GEMM tests ---

void test_tiled_matches_naive() {
    // Deliberately non-power-of-2 sizes to stress edge handling
    size_t sizes[] = {7, 15, 16, 17, 31, 32, 33, 64, 100};
    size_t block_sizes[] = {8, 16, 32};

    for (size_t n : sizes) {
        Tensor A(n, n), B(n, n);
        A.randomize(300 + static_cast<unsigned>(n));
        B.randomize(400 + static_cast<unsigned>(n));

        Tensor C_naive(n, n), C_tiled(n, n);
        gemm_naive(A, B, C_naive);

        for (size_t bs : block_sizes) {
            C_tiled.zero();
            gemm_tiled(A, B, C_tiled, bs);

            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j)
                    ASSERT_NEAR(C_tiled.at(i,j), C_naive.at(i,j));
        }
    }
}

void test_tiled_block_size_1() {
    // block_size=1 degenerates to element-by-element — same result as naive
    Tensor A(8, 8), B(8, 8), C_naive(8, 8), C_tiled(8, 8);
    A.randomize(500);
    B.randomize(600);

    gemm_naive(A, B, C_naive);
    gemm_tiled(A, B, C_tiled, 1);

    for (size_t i = 0; i < 8; ++i)
        for (size_t j = 0; j < 8; ++j)
            ASSERT_NEAR(C_tiled.at(i,j), C_naive.at(i,j));
}

void test_tiled_block_size_ge_n() {
    // block_size >= N degenerates to a single tile — same as naive
    Tensor A(10, 10), B(10, 10), C_naive(10, 10), C_tiled(10, 10);
    A.randomize(700);
    B.randomize(800);

    gemm_naive(A, B, C_naive);
    gemm_tiled(A, B, C_tiled, 256);

    for (size_t i = 0; i < 10; ++i)
        for (size_t j = 0; j < 10; ++j)
            ASSERT_NEAR(C_tiled.at(i,j), C_naive.at(i,j));
}

void test_tiled_dimension_mismatch() {
    Tensor A(2, 3), B(4, 2), C(2, 2);
    ASSERT_THROWS(gemm_tiled(A, B, C, 16), std::invalid_argument);
}

int main() {
    std::cout << "test_gemm (naive):" << std::endl;
    RUN_TEST(test_2x2_known);
    RUN_TEST(test_rectangular_3x2_times_2x4);
    RUN_TEST(test_identity_multiplication);
    RUN_TEST(test_zero_matrix);
    RUN_TEST(test_dimension_mismatch);
    RUN_TEST(test_random_vs_reference);
    RUN_TEST(test_1x1);

    std::cout << "test_gemm (tiled):" << std::endl;
    RUN_TEST(test_tiled_matches_naive);
    RUN_TEST(test_tiled_block_size_1);
    RUN_TEST(test_tiled_block_size_ge_n);
    RUN_TEST(test_tiled_dimension_mismatch);

    TEST_SUMMARY("test_gemm");
}
