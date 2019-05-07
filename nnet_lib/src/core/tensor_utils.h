#ifndef _CORE_TENSOR_UTILS_H_
#define _CORE_TENSOR_UTILS_H_

#include <iostream>
#include <vector>

#include "core/tensor.h"

namespace smaug {

class Workspace;

std::ostream& operator<<(std::ostream& os, const TensorIndexIterator& iter);
std::ostream& operator<<(std::ostream& os, const TensorShape& shape);
std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

template <typename DType>
void printTensorElement(std::ostream& os, DType* data, int index) {
    os << data[index];
}

template <>
void printTensorElement<float16>(std::ostream& os, float16* data, int index);

template <typename DType>
void writeTensorToOstream(std::ostream& os, const Tensor& tensor) {
    const TensorShape& shape = tensor.getShape();
    if (shape.ndims() == 0) {
        os << "  [ ]\n";
        return;
    }
    int ndims = shape.ndims();
    int newlineAfterElems = shape[ndims - 1];
    int newGroupAfterElems =
            (shape.ndims() >= 2 ? shape[ndims - 1] * shape[ndims - 2]
                                : shape[ndims - 1]);
    int counter = 0;
    DType* data = tensor.template data<DType>();
    os << tensor.getName() << ", shape = " << shape << "\n";
    for (auto idx = tensor.startIndex(); !idx.end(); ++idx) {
        // Print the current index after going through all of the last two
        // dimensions.
        if (counter == 0)
            os << idx << "\n[ ";
        printTensorElement<DType>(os, data, idx);
        os << " ";
        ++counter;
        if (counter % newGroupAfterElems == 0) {
            counter = 0;
            os << " ]\n";
        } else if (counter % newlineAfterElems == 0) {
            os << "\n  ";
        }
    }
}

namespace internal {
template <typename DType>
void copyTensorRegion(Tensor* dest,
                      Tensor* src,
                      const std::vector<int>& destOrigin,
                      const std::vector<int>& srcOrigin,
                      const std::vector<int>& regionSize) {
    auto destIt =
            TensorRegionIndexIterator(dest->getShape(), destOrigin, regionSize);
    auto srcIt =
            TensorRegionIndexIterator(src->getShape(), srcOrigin, regionSize);
    DType* destPtr = dest->template data<DType>();
    DType* srcPtr = src->template data<DType>();
    while (!srcIt.end() && !destIt.end()) {
        destPtr[destIt] = srcPtr[srcIt];
        ++destIt;
        ++srcIt;
    }
}
}  // namespace internal

void copyTensorRegion(Tensor* dest,
                      Tensor* src,
                      std::vector<int> destOrigin,
                      std::vector<int> srcOrigin,
                      std::vector<int> regionSize);

// This generates a TiledTensor from a Tensor using the specified tile shape.
TiledTensor generateTiledTensor(Tensor* tensor,
                                const TensorShape& tileShape,
                                std::vector<int> halos,
                                Workspace* workspace);

// This will copy data from a tiled tensor into a single tensor. We name it as
// "untile" because what it does reverses the tiling process.
void untileTiledTensor(TiledTensor& tiledTensor, Tensor* destTensor);

// This concatenates tensors on the specified dimension into one single tensor.
Tensor* concatTensors(std::vector<Tensor*> inputTensors,
                      int concatDim,
                      Workspace* workspace);

}  // namespace smaug

#endif
