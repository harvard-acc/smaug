#include <iostream>

#include "fp16.h"
#include "smaug/core/workspace.h"
#include "smaug/core/tensor.h"
#include "smaug/core/tensor_utils.h"
#include "smaug/utility/debug_stream.h"

namespace smaug {

template <>
void printTensorElement<float16>(std::ostream& os, const float16* data, int index) {
    os << fp16_ieee_to_fp32_value(data[index]);
}

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
        case Bool:
            writeTensorToOstream<bool>(os, tensor);
            break;
        default:
            assert(false && "Unknown data type!");
    }
    return os;
}

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
        case Bool:
            internal::copyTensorRegion<bool>(
                    dest, src, destOrigin, srcOrigin, regionSize);
            break;
        default:
            assert(false && "Unknown data type!");
    }
}

void copyTensorData(Tensor* dest,
                    Tensor* src,
                    std::vector<int> destOrigin,
                    std::vector<int> srcOrigin,
                    int copySize) {
    assert(dest->getDataType() == src->getDataType());
    switch (dest->getDataType()) {
        case Float16:
            internal::copyTensorData<uint16_t>(
                    dest, src, destOrigin, srcOrigin, copySize);
            break;
        case Float32:
            internal::copyTensorData<float>(
                    dest, src, destOrigin, srcOrigin, copySize);
            break;
        case Float64:
            internal::copyTensorData<double>(
                    dest, src, destOrigin, srcOrigin, copySize);
            break;
        case Int32:
            internal::copyTensorData<int>(
                    dest, src, destOrigin, srcOrigin, copySize);
            break;
        case Int64:
            internal::copyTensorData<int64_t>(
                    dest, src, destOrigin, srcOrigin, copySize);
            break;
        case Bool:
            internal::copyTensorData<bool>(
                    dest, src, destOrigin, srcOrigin, copySize);
            break;
        default:
            assert(false && "Unknown data type!");
    }
}

void copyRawTensorData(Tensor* dest,
                       Tensor* src,
                       int destOffset,
                       int srcOffset,
                       int copySize) {
    assert(dest->getDataType() == src->getDataType());
    switch (dest->getDataType()) {
        case Float16:
            internal::copyRawTensorData<uint16_t>(
                    dest, src, destOffset, srcOffset, copySize);
            break;
        case Float32:
            internal::copyRawTensorData<float>(
                    dest, src, destOffset, srcOffset, copySize);
            break;
        case Float64:
            internal::copyRawTensorData<double>(
                    dest, src, destOffset, srcOffset, copySize);
            break;
        case Int32:
            internal::copyRawTensorData<int>(
                    dest, src, destOffset, srcOffset, copySize);
            break;
        case Int64:
            internal::copyRawTensorData<int64_t>(
                    dest, src, destOffset, srcOffset, copySize);
            break;
        case Bool:
            internal::copyRawTensorData<bool>(
                    dest, src, destOffset, srcOffset, copySize);
            break;
        default:
            assert(false && "Unknown data type!");
    }
}

namespace internal {
// Compute the tile size in this dimension with padding accounted for. The goal
// is to get the tile dimension size that doesn't have any elements unused,
// given the padding, weight and stride sizes.
//
// Args:
//   maxTileDim: Maximum size of the tile size in this dimension.
//   padding: Padding in the dimension.
//   weightDim: Weight size in this dimension.
//   stride: Stride size in this dimension.
// Returns:
//   The tile size in this dimension.
int computePaddedTileDim(int maxTileDim,
                         int padding,
                         int weightDim,
                         int stride) {
    // The number of strides we can take in this dimension.
    int numStrides = (maxTileDim + padding - weightDim) / stride;
    if (numStrides <= 0)
        return maxTileDim;
    int tileDim = weightDim + stride * numStrides;
    return tileDim - padding;
}
}  // namespace internal

TiledTensor generateTiledTensor(Tensor* tensor,
                                const TensorShape& tileShape,
                                Operator* op,
                                int fieldRows,
                                int fieldCols,
                                int rowStride,
                                int colStride,
                                PaddingType paddingType) {
    const TensorShape& inputShape = tensor->getShape();
    const int ndims = inputShape.ndims();
    DataLayout layout = inputShape.getLayout();
    // Compute the tiling halos. These are the rows/columns that the subsequent
    // tile will overlap with the previous tile.
    std::vector<int> tilingHalos(ndims, 0);
    int hIdx = layout == NHWC ? 1 : NCHW ? 2 : -1;
    int wIdx = layout == NHWC ? 2 : NCHW ? 3 : -1;
    // The tilingHalos could be negative if fieldRows < rowStride, but we
    // actually want that. For example, fieldRows = 1, rowStride = 2, then what
    // the next tile wants is not to "borrow" any rows from the previous tile,
    // but skip one row and start from there. So, -1 actually gives us the
    // skipping effect.
    if (hIdx != -1 && fieldRows != 0)
        tilingHalos[hIdx] = fieldRows - rowStride;
    if (wIdx != -1 && fieldCols != 0)
        tilingHalos[wIdx] = fieldCols - colStride;
    // Compute the input paddings.
    int totalRowPad = (paddingType == SamePadding) ? fieldRows - 1 : 0;
    int totalColPad = (paddingType == SamePadding) ? fieldCols - 1 : 0;
    int topPad = FRAC_CEIL(totalRowPad, 2);
    int leftPad = FRAC_CEIL(totalColPad, 2);
    // This contains tile shapes in each dimension.
    std::vector<std::vector<int>> tilesInDim(ndims);
    // Compute the tile shapes in each dimension.
    for (int i = 0; i < ndims; i++) {
        int remaining = inputShape[i];
        while (remaining > 0) {
            int tileDim = std::min(tileShape[i], remaining);
            bool firstTileInDim = tilesInDim[i].size() == 0;
            bool lastTileInDim = remaining <= tileShape[i];
            // Adjust the tile dimension size if we are at the first tile
            // because of the top/left paddings.
            if (i == hIdx && firstTileInDim && !lastTileInDim) {
                tileDim = internal::computePaddedTileDim(
                        tileDim, topPad, fieldRows, rowStride);
            } else if (i == wIdx && firstTileInDim && !lastTileInDim) {
                tileDim = internal::computePaddedTileDim(
                        tileDim, leftPad, fieldCols, colStride);
            }
            tilesInDim[i].push_back(tileDim);
            remaining -= tileDim;
            if (remaining > 0)
                remaining += tilingHalos[i];
        }
    }
    std::vector<int> numBlocksInDim(ndims, 0);
    for (int i = 0; i < ndims; i++)
        numBlocksInDim[i] = tilesInDim[i].size();
    TiledTensor tiledTensor(
            TensorShape(numBlocksInDim, inputShape.getLayout()), tensor);
    if (tiledTensor.size() == 1) {
        // If there's only one tile, we don't need to tile the original tensor.
        // So directly use it as the tile.
        tiledTensor[0] = tensor;
    } else {
        std::vector<int> currentOrigin(ndims, 0);
        for (auto tileIndex = tiledTensor.startIndex(); !tileIndex.end();
             ++tileIndex) {
            std::vector<int> currentTileShape(ndims);
            for (int i = 0; i < ndims; i++)
                currentTileShape[i] = tilesInDim[i][tileIndex.currentIndex(i)];
            TensorShape currentShape(currentTileShape,
                                     tileShape.getLayout(),
                                     tileShape.getAlignment());
            std::string tileName = op->getName() + ":" + tensor->getName() +
                                   "/tile:" + std::to_string((int)tileIndex);
            Tensor* tile = new Tensor(tileName, currentShape);
            tile->allocateStorage(tensor->getDataType());
            tiledTensor.setTile(tileIndex, currentOrigin, tile, false);
            for (int i = ndims - 1; i >= 0; i--) {
                currentOrigin[i] += currentShape[i];
                if (currentOrigin[i] >= inputShape[i]) {
                    currentOrigin[i] = 0;
                } else {
                    currentOrigin[i] -= tilingHalos[i];
                    break;
                }
            }
        }
    }
    op->getWorkspace()->addTiledTensor(tiledTensor);
    dout(1) << "  Tiled Tensor " << tensor->getName() << ":\n"
            << "    original tensor shape: " << tensor->getShape() << "\n"
            << "    tile shape: " << tileShape
            << ", number of tiles: " << tiledTensor.size() << "\n";
    return tiledTensor;
}

void flattenTiledTensor(TiledTensor& tiledTensor, Tensor* destTensor) {
    const TensorShape& tensorShape = destTensor->getShape();
    int ndims = tensorShape.ndims();
    int destOffset = 0;
    for (auto tileIndex = tiledTensor.startIndex(); !tileIndex.end();
         ++tileIndex) {
        Tensor* tile = tiledTensor[tileIndex];
        const TensorShape& tileShape = tile->getShape();
        copyRawTensorData(
                destTensor, tile, destOffset, 0, tileShape.storageSize());
        destOffset += tileShape.storageSize();
    }
}

Tensor* concatTensors(std::vector<Tensor*> inputTensors,
                      int concatDim,
                      Workspace* workspace) {
    std::string outputName = inputTensors[0]->getName();
    TensorShape inputShape = inputTensors[0]->getShape();
    std::vector<int> outputDims = inputShape.dims();
    // Calculate the shape for the output tensor.
    for (int i = 1; i < inputTensors.size(); i++) {
        outputName += ("-" + inputTensors[i]->getName());
        outputDims[concatDim] += inputTensors[i]->getShape()[concatDim];
    }
    TensorShape outputShape(
            outputDims, inputShape.getLayout(), inputShape.getAlignment());
    Tensor* outputTensor = new Tensor(outputName, outputShape);
    workspace->addTensor(outputTensor);
    outputTensor->allocateStorage(inputTensors[0]->getDataType());
    // Copy data into the output tensor.
    int ndims = inputShape.ndims();
    std::vector<int> currentOrigin(ndims, 0);
    std::vector<int> srcOrigin(ndims, 0);
    for (int i = 0; i < inputTensors.size(); i++) {
        TensorShape srcShape = inputTensors[i]->getShape();
        copyTensorRegion(outputTensor,
                         inputTensors[i],
                         currentOrigin,
                         srcOrigin,
                         srcShape.dims());
        currentOrigin[concatDim] += srcShape[concatDim];
    }
    return outputTensor;
}

}  // namespace smaug
