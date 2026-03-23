#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>

namespace tile_runtime {

template <typename T, std::size_t Alignment = 64>
class AlignedAllocator {
public:
    using value_type = T;

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        std::size_t bytes = n * sizeof(T);
        // std::aligned_alloc requires size to be a multiple of alignment
        bytes = ((bytes + Alignment - 1) / Alignment) * Alignment;
        void* ptr = std::aligned_alloc(Alignment, bytes);
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, std::size_t) noexcept {
        std::free(ptr);
    }

    template <typename U>
    struct rebind { using other = AlignedAllocator<U, Alignment>; };

    template <typename U>
    bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept { return true; }
    template <typename U>
    bool operator!=(const AlignedAllocator<U, Alignment>&) const noexcept { return false; }
};

}  // namespace tile_runtime
