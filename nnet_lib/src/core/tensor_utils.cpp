#include <iostream>

#include "fp16.h"
#include "core/workspace.h"
#include "core/tensor.h"
#include "core/tensor_utils.h"
#include "utility/debug_stream.h"

namespace smaug {

template <>
void printTensorElement<float16>(std::ostream& os, float16* data, int index) {
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

// Copy a block of a tensor to another tensor.
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
        default:
            assert(false && "Unknown data type!");
    }
}

TiledTensor generateTiledTensor(Tensor* tensor,
                                const TensorShape& tileShape,
                                std::vector<int> halos,
                                Operator* op) {
    assert(halos.size() == tileShape.ndims());
    const TensorShape& inputShape = tensor->getShape();
    const int ndims = inputShape.ndims();
    std::vector<int> numBlocksInDim(ndims, 0);
    for (int i = 0; i < ndims; i++) {
        int remaining = inputShape[i];
        while (remaining > 0) {
            numBlocksInDim[i]++;
            remaining -= tileShape[i];
            if (remaining > 0)
                remaining += halos[i];
        }
    }
    TiledTensor tiledTensor(
            TensorShape(numBlocksInDim, inputShape.getLayout()));
    std::vector<int> currentOrigin(ndims, 0);
    std::vector<int> dstOrigin(ndims, 0);
    for (auto tileIndex = tiledTensor.startIndex(); !tileIndex.end();
         ++tileIndex) {
        std::vector<int> currentTileShape(ndims);
        for (int i = 0; i < ndims; i++) {
            currentTileShape[i] =
                    std::min(inputShape[i] - currentOrigin[i], tileShape[i]);
        }
        TensorShape currentShape(currentTileShape,
                                 tileShape.getLayout(),
                                 tileShape.getAlignment());
        std::string tileName = op->getName() + ":" + tensor->getName() +
                               "/tile:" + std::to_string((int)tileIndex);
        Tensor* tile = new Tensor(tileName, currentShape);
        tile->allocateStorage(tensor->getDataType());
        copyTensorRegion(tile,
                         tensor,
                         dstOrigin,
                         currentOrigin,
                         currentShape.dims());
        for (int i = ndims - 1; i >= 0; i--) {
            currentOrigin[i] += currentShape[i];
            if (currentOrigin[i] >= inputShape[i]) {
                currentOrigin[i] = 0;
            } else {
                currentOrigin[i] -= halos[i];
                break;
            }
        }
        tiledTensor[tileIndex] = tile;
    }
    op->getWorkspace()->addTiledTensor(tiledTensor);
    dout(1) << "Tiled Tensor " << tensor->getName() << ": \n"
            << "  tile shape: " << tileShape
            << ", number of tiles: " << tiledTensor.size() << "\n";
    return tiledTensor;
}

void untileTiledTensor(TiledTensor& tiledTensor, Tensor* destTensor) {
    const TensorShape& tensorShape = destTensor->getShape();
    int ndims = tensorShape.ndims();
    std::vector<int> currentOrigin(ndims, 0);
    std::vector<int> srcOrigin(ndims, 0);
    for (auto tileIndex = tiledTensor.startIndex(); !tileIndex.end();
         ++tileIndex) {
        Tensor* tile = tiledTensor[tileIndex];
        const TensorShape& tileShape = tile->getShape();
        copyTensorRegion(destTensor,
                         tile,
                         currentOrigin,
                         srcOrigin,
                         tileShape.dims());
        for (int i = ndims - 1; i >= 0; i--) {
            currentOrigin[i] += tileShape[i];
            if (currentOrigin[i] >= tensorShape[i]) {
                currentOrigin[i] = 0;
            } else
                break;
        }
    }
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
