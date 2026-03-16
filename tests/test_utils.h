#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

static int g_test_failures = 0;

#define ASSERT_TRUE(expr)                                                  \
    do {                                                                   \
        if (!(expr)) {                                                     \
            std::cerr << "FAIL: " << __FILE__ << ":" << __LINE__           \
                      << "  " #expr << std::endl;                          \
            ++g_test_failures;                                             \
        }                                                                  \
    } while (0)

#define ASSERT_EQ(a, b)                                                    \
    do {                                                                   \
        if ((a) != (b)) {                                                  \
            std::cerr << "FAIL: " << __FILE__ << ":" << __LINE__           \
                      << "  " #a " == " #b << "  (" << (a)                 \
                      << " vs " << (b) << ")" << std::endl;               \
            ++g_test_failures;                                             \
        }                                                                  \
    } while (0)

constexpr float TEST_TOLERANCE = 1e-4f;

#define ASSERT_NEAR(a, b)                                                  \
    do {                                                                   \
        float _a = (a), _b = (b);                                         \
        float _diff = std::fabs(_a - _b);                                  \
        float _scale = std::fmax(1.0f, std::fmax(std::fabs(_a),            \
                                                  std::fabs(_b)));         \
        if (_diff / _scale > TEST_TOLERANCE) {                             \
            std::cerr << "FAIL: " << __FILE__ << ":" << __LINE__           \
                      << "  " #a " ~= " #b << "  (" << _a                 \
                      << " vs " << _b << ", diff=" << _diff << ")"         \
                      << std::endl;                                        \
            ++g_test_failures;                                             \
        }                                                                  \
    } while (0)

#define ASSERT_THROWS(expr, exception_type)                                \
    do {                                                                   \
        bool _caught = false;                                              \
        try { expr; } catch (const exception_type&) { _caught = true; }    \
        if (!_caught) {                                                    \
            std::cerr << "FAIL: " << __FILE__ << ":" << __LINE__           \
                      << "  expected " #exception_type " from " #expr      \
                      << std::endl;                                        \
            ++g_test_failures;                                             \
        }                                                                  \
    } while (0)

#define RUN_TEST(fn)                                                       \
    do {                                                                   \
        std::cout << "  " #fn "... ";                                      \
        int _before = g_test_failures;                                     \
        fn();                                                              \
        std::cout << (g_test_failures == _before ? "OK" : "FAILED")        \
                  << std::endl;                                            \
    } while (0)

#define TEST_SUMMARY(suite)                                                \
    do {                                                                   \
        if (g_test_failures == 0)                                          \
            std::cout << suite ": all tests passed." << std::endl;         \
        else                                                               \
            std::cerr << suite ": " << g_test_failures                     \
                      << " failure(s)." << std::endl;                      \
        return g_test_failures == 0 ? 0 : 1;                              \
    } while (0)
