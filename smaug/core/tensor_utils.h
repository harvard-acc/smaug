/**
 * \file tensor_utils.h
 * \brief Utility functions for copying/printing/tiling tensors.
 */

#ifndef _CORE_TENSOR_UTILS_H_
#define _CORE_TENSOR_UTILS_H_

#include <iostream>
#include <vector>
#include <cstring>

#include "smaug/core/tensor.h"

namespace smaug {

class Workspace;
class Operator;

std::ostream& operator<<(std::ostream& os, const TensorIndexIterator& iter);
std::ostream& operator<<(std::ostream& os, const TensorShape& shape);
std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

template <typename DType>
void printTensorElement(std::ostream& os, const DType* data, int index) {
    os << data[index];
}

template <>
void printTensorElement<float16>(std::ostream& os, const float16* data, int index);

/**
 * Pretty-print a Tensor's name, shape, and contents to the provided ostream.
 */
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
    const DType* data = tensor.template data<DType>();
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
    const TensorShape& srcShape = src->getShape();
    const TensorShape& destShape = dest->getShape();
    TensorShape regionShape(
            regionSize, srcShape.getLayout(), srcShape.getAlignment());
    const int ndims = srcShape.ndims();
    auto destIt = TensorRegionIndexIterator(destShape, destOrigin, regionSize);
    auto srcIt = TensorRegionIndexIterator(srcShape, srcOrigin, regionSize);

    // We know where to copy data from and how much data we should copy (the
    // data region), now starting from the last dimension, we figure out how
    // much contiguous data there exists such that we can apply more efficient
    // data copy mechanisms (memcpy).
    std::vector<int> contiguousRegion(ndims, 1);
    int contiguousSize = 1;
    for (int i = ndims - 1; i >= 0; i--) {
        contiguousSize *= regionShape.getStorageDim(i);
        contiguousRegion[i] = regionShape[i];
        // If we find a region dimension smaller than that of either src or
        // dest tensor, then the next region dimension must not be contiguous.
        if (regionShape[i] < srcShape[i] || regionShape[i] < destShape[i])
            break;
    }

    // Copy the data region from the src tensor to the dest tensor.
    DType* destPtr = dest->template data<DType>();
    DType* srcPtr = src->template data<DType>();
    while (!srcIt.end() && !destIt.end()) {
#ifdef PEDANTIC
        destPtr[destIt] = srcPtr[srcIt];
        ++destIt;
        ++srcIt;
#else
        memcpy(&destPtr[destIt],
               &srcPtr[srcIt],
               contiguousSize * sizeof(DType));
        destIt += contiguousRegion;
        srcIt += contiguousRegion;
#endif
    }
}

template <typename DType>
void copyRawTensorData(Tensor* dest,
                       Tensor* src,
                       int destOffset,
                       int srcOffset,
                       int copySize) {
    DType* destPtr = dest->template data<DType>();
    DType* srcPtr = src->template data<DType>();
    std::memcpy(
            &destPtr[destOffset], &srcPtr[srcOffset], copySize * sizeof(DType));
}

template <typename DType>
void copyTensorData(Tensor* dest,
                    Tensor* src,
                    std::vector<int> destOrigin,
                    std::vector<int> srcOrigin,
                    int copySize) {
    TensorIndexIterator destIdx = dest->startIndex();
    TensorIndexIterator srcIdx = src->startIndex();
    destIdx += destOrigin;
    srcIdx += srcOrigin;
    DType* destPtr = dest->template data<DType>();
    DType* srcPtr = src->template data<DType>();
    for (; !srcIdx.end(); ++srcIdx, ++destIdx)
        destPtr[destIdx] = srcPtr[srcIdx];
}

}  // namespace internal

// Copy a region of data from one tensor to another.
//
/**
 * Copies a region of a source Tensor to a corresponding region in a
 * destination Tensor. The two Tensors are expected to share the same layout.
 * Region origins and sizes are all specified in elements (not bytes) and in
 * accordance with the data layout.
 *
 * For example:
 *   `tensorA`: 4x4, tensor B: 3x3
 *   To copy upper left 2x2 block of `tensorA` to the lower left 2x2 block of
 `*   tensorB`:
 *      `copyTensorRegion(tensorB, tensorA, {1,1}, {0,0}, {2,2})`
 *
 * @param dest Destination Tensor
 * @param src Source Tensor
 * @param destOrigin The start of the copied region in the destination.
 * @param srcOrigin The start of the copied region in the source.
 * @param regionSize The size of the region.
 */
void copyTensorRegion(Tensor* dest,
                      Tensor* src,
                      std::vector<int> destOrigin,
                      std::vector<int> srcOrigin,
                      std::vector<int> regionSize);

/**
 * Similar to copyTensorRegion, but the region is a contiguous block of
 * memory.
 */
void copyTensorData(Tensor* dest,
                    Tensor* src,
                    std::vector<int> destOffset,
                    std::vector<int> srcOffset,
                    int copySize);

/**
 * Directly copies a linear region of memory from dest to src, without taking
 * dimensions/padding into account.
 *
 * @param dest Destination Tensor
 * @param src Source Tensor
 * @param destOffset The linear offset into the destination where data will be
 * copied to.
 * @param srcOffset The linear offset into the source where data will be copied
 * from.
 * @param copySize The size of the region in elements.
 */
void copyRawTensorData(
        Tensor* dest, Tensor* src, int destOffset, int srcOffset, int copySize);


/**
 * Generates a TiledTensor from a source Tensor with the specified tile shape.
 *
 * Depending on the operator that needs this TiledTensor, tiles may need to
 * overlap each other (e.g. for a convolutional filter window).
 *
 * @param tensor The Tensor to tile.
 * @param tileShape The maximum size of each tile.
 * @param op The Operator that will be consuming this TiledTensor.
 * @param fieldRows Number of rows of a filter applied, if any.
 * @param fieldCols Number of columns of a filter applied, if any.
 * @param rowStride The row stride of a filter applied, if any.
 * @param colStride The column stride of a filter applied, if any.
 * @param paddingType The type of additional zero-padding applied on the Tensor
 * by the Operator, if any.
 */
TiledTensor generateTiledTensor(Tensor* tensor,
                                const TensorShape& tileShape,
                                Operator* op,
                                int fieldRows = 0,
                                int fieldCols = 0,
                                int rowStride = 1,
                                int colStride = 1,
                                PaddingType paddingType = ValidPadding);

/**
 * A helper method to both tile a Tensor and fill the tiles with data.
 */
TiledTensor generateTiledTensorAndCopyData(
        Tensor* tensor,
        const TensorShape& tileShape,
        Operator* op,
        int fieldRows = 0,
        int fieldCols = 0,
        int rowStride = 1,
        int colStride = 1,
        PaddingType paddingType = ValidPadding);

/**
 * A helper method to both tile a Tensor and fill the tiles with data.
 */
template <typename... Args>
TiledTensor generateTiledTensorAndCopyData(Args&&... args) {
    TiledTensor tiledTensor =
            generateTiledTensor(std::forward<Args>(args)...);
    tiledTensor.copyDataToAllTiles();
    return tiledTensor;
}

/**
 * Copies the data from each tile in a TiledTensor into a destination Tensor as
 * a contiguous block of memory, as if only one dimension ever existed.
 */
void flattenTiledTensor(TiledTensor& tiledTensor, Tensor* destTensor);

/**
 * Concatenates Tensors on the specified dimension into one single tensor.
 */
Tensor* concatTensors(std::vector<Tensor*> inputTensors,
                      int concatDim,
                      Workspace* workspace);

}  // namespace smaug

#endif
