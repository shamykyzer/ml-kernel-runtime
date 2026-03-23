#pragma once

#include "aligned_allocator.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace tile_runtime {

class Tensor {
public:
    Tensor();
    Tensor(size_t rows, size_t cols);
    Tensor(size_t rows, size_t cols, float init_value);

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return data_.size(); }

    float& at(size_t i, size_t j);
    const float& at(size_t i, size_t j) const;

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    void fill(float value);
    void zero();
    void randomize(unsigned int seed);

private:
    size_t rows_;
    size_t cols_;
    std::vector<float, AlignedAllocator<float, 64>> data_;

    void bounds_check(size_t i, size_t j) const;
};

}  // namespace tile_runtime
