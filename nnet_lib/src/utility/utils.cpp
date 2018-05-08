#include <cassert>

#include "core/datatypes.h"
#include "utility/utils.h"

namespace smaug {

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
