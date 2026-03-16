#include "tensor.h"

#include <random>
#include <sstream>

namespace tile_runtime {

Tensor::Tensor() : rows_(0), cols_(0) {}

Tensor::Tensor(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0f) {}

Tensor::Tensor(size_t rows, size_t cols, float init_value)
    : rows_(rows), cols_(cols), data_(rows * cols, init_value) {}

void Tensor::bounds_check(size_t i, size_t j) const {
    if (i >= rows_ || j >= cols_) {
        std::ostringstream oss;
        oss << "Tensor index (" << i << ", " << j
            << ") out of range for shape (" << rows_ << ", " << cols_ << ")";
        throw std::out_of_range(oss.str());
    }
}

float& Tensor::at(size_t i, size_t j) {
    bounds_check(i, j);
    return data_[i * cols_ + j];
}

const float& Tensor::at(size_t i, size_t j) const {
    bounds_check(i, j);
    return data_[i * cols_ + j];
}

void Tensor::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

void Tensor::zero() {
    fill(0.0f);
}

void Tensor::randomize(unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : data_) {
        v = dist(gen);
    }
}

}  // namespace tile_runtime
