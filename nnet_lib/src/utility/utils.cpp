#include <cassert>

#include "core/datatypes.h"
#include "operators/common.h"
#include "utility/utils.h"

namespace smaug {

void* malloc_aligned(size_t size) {
    void* ptr = NULL;
    int err = posix_memalign(
            (void**)&ptr, CACHELINE_SIZE, next_multiple(size, CACHELINE_SIZE));
    assert(err == 0 && "Failed to allocate memory!");
    return ptr;
}

std::string dataLayoutToStr(DataLayout layout) {
    switch (layout) {
        case DataLayout::NCHW:
            return "NCHW";
        case DataLayout::NHWC:
            return "NHWC";
        case DataLayout::NC:
            return "NC";
        case DataLayout::X:
            return "X";
        default:
            assert(false && "Unknown data layout!");
            return "";
    }
}

int calc_padding(int value, unsigned alignment) {
    if (alignment == 0 || value % alignment == 0)
        return 0;
    return (alignment - (value % alignment));
}

}  // namespace smaug
