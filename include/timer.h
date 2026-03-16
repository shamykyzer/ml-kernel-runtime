#pragma once

#include <chrono>

namespace tile_runtime {

class Timer {
public:
    void start() { start_ = clock::now(); }
    void stop() { end_ = clock::now(); }

    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(end_ - start_).count();
    }

    double elapsed_sec() const {
        return std::chrono::duration<double>(end_ - start_).count();
    }

private:
    using clock = std::chrono::high_resolution_clock;
    clock::time_point start_;
    clock::time_point end_;
};

}  // namespace tile_runtime
