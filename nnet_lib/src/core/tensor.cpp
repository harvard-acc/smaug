#include <iostream>

#include "core/tensor.h"

namespace smaug {

std::ostream& operator<<(std::ostream& os, const TensorShape& shape) {
    os << "(";
    for (int i = 0; i < shape.ndims(); i++) {
        os << shape[i];
        if (i != shape.ndims() - 1)
            os << ", ";
    }
    os << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const TensorIndexIterator& iter) {
    os << "( ";
    for (int i = 0; i < iter.dims.size(); ++i) {
        os << iter.state[i] << " ";
    }
    os << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    DataType type = tensor.getDataType();
    switch (type) {
        case Float16:
            writeTensorToOstream<uint16_t>(os, tensor);
            break;
        case Float32:
            writeTensorToOstream<float>(os, tensor);
            break;
        case Float64:
            writeTensorToOstream<double>(os, tensor);
            break;
        case Int32:
            writeTensorToOstream<int>(os, tensor);
            break;
        case Int64:
            writeTensorToOstream<int64_t>(os, tensor);
            break;
        default:
            assert(false && "Unknown data type!");
    }
    return os;
}

// Copy a block of a tensor to another tensor.
//
// For example:
//   tensor A: 4x4, tensor B: 3x3
//   To copy upper left 2x2 block of tensor A to the lower left 2x2 block of
//   tensor B:
//      copyTensorRegion(tensorB, tensorA, {1,1}, {0,0}, {2,2});
void copyTensorRegion(Tensor* dest,
                      Tensor* src,
                      std::vector<int> destOrigin,
                      std::vector<int> srcOrigin,
                      std::vector<int> regionSize) {
    assert(dest->ndims() == src->ndims());
    assert(dest->getDataType() == src->getDataType());
    switch (dest->getDataType()) {
        case Float16:
            internal::copyTensorRegion<uint16_t>(
                    dest, src, destOrigin, srcOrigin, regionSize);
            break;
        case Float32:
            internal::copyTensorRegion<float>(
                    dest, src, destOrigin, srcOrigin, regionSize);
            break;
        case Float64:
            internal::copyTensorRegion<double>(
                    dest, src, destOrigin, srcOrigin, regionSize);
            break;
        case Int32:
            internal::copyTensorRegion<int>(
                    dest, src, destOrigin, srcOrigin, regionSize);
            break;
        case Int64:
            internal::copyTensorRegion<int64_t>(
                    dest, src, destOrigin, srcOrigin, regionSize);
            break;
        default:
            assert(false && "Unknown data type!");
    }
}

}  // namespace smaug
