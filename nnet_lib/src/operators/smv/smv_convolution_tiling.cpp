#include <algorithm>

#include "core/backend.h"
#include "operators/common.h"
#include "operators/smv/smv_convolution_op.h"
#include "operators/smv/smv_convolution_tiling.h"
#include "utility/debug_stream.h"

namespace smaug {
namespace smv {
namespace conv {

std::ostream& operator<<(std::ostream& os, const TilingDims& dims) {
  switch (dims) {
      case None:
          os << "None";
          break;
      case DimN:
          os << "DimN";
          break;
      case DimNC:
          os << "DimNC";
          break;
      case DimNH:
          os << "DimNH";
          break;
      case Invalid:
          os << "Invalid";
          break;
  }
  return os;
}

// Find the best set of dimensions to tile a given tensor shape.
//
// The goal is to divide up a tensor into tiles that each are <= maxTileSize
// elements.  Assuming that the input tensor data layout is NHWC, the
// preferences for tiling dimensions are as follows:
//   1) No tiling.
//   2) Dim-N tiling. For inputs/outputs, this would be over batches of inputs.
//      For weights, this would be over filters/output channels.
//   3) Dim-NC tiling. After tiling by N, tile channelwise. Do not tile in HW.
//   4) Dim-NH tiling. After tiling by N, tile rowwise. Do not tile in WC.
//
// For options 2-4, a minimum size for each dimension can be specified via
// minN, minH, and minC.
TilingDims TilingOptimizer::findBestTilingDims(const TensorShape& shape,
                                               int maxTileSize,
                                               int minN,
                                               int minH,
                                               int minC) {
    minN = std::min(shape[0], minN);
    minH = std::min(shape[1], minH);
    // C is the last dimension, so we need to apply padding.
    minC = std::min(shape[3] + shape.getPadding(3), minC);
    if (shape.storageSize() <= maxTileSize)
        return TilingDims::None;
    int sizePerN = shape.storageSize() / shape[0];
    if (sizePerN * minN <= maxTileSize)
        return TilingDims::DimN;
    if (sizePerN / shape[3] * minC <= maxTileSize)
        return TilingDims::DimNC;
    if (sizePerN / shape[1] * minH <= maxTileSize)
        return TilingDims::DimNH;
    std::cerr << "[ERROR]: Unable to find a supported set of tiling dimensions "
                 "for tensor with shape " << shape << "!\n";
    assert(false && "Unable to find valid tiling dimensions.");
    return TilingDims::Invalid;
}

// Determine the best tiling dimensions for running convolution on SMV.
//
// This function imposes some additional constraints on the tiling dimensions,
// in that certain combinations of input/weight/output tiling dimensions are
// not allowed in the interest of tiling code complexity.
//
// Returns:
//   A 3-element array of TilingDims enums (inputs, weights, outputs).
std::array<TilingDims, 3> TilingOptimizer::determineBestTilingDims(
        Tensor* inputs, Tensor* weights, Tensor* outputs, int maxTileSize) {
    std::vector<TilingConfig> allTilingConfigs;
    // Determine the best tiling strategy for each of inputs, weights, and
    // outputs. Don't try to figure out the actual tile sizes yet.
    TilingDims bestInputTilingDims = findBestTilingDims(inputs->getShape(),
                                                        maxTileSize,
                                                        1,
                                                        weights->getShape()[1],
                                                        kNumMaccsPerPE);
    TilingDims bestWeightTilingDims = findBestTilingDims(weights->getShape(),
                                                         maxTileSize,
                                                         kNumPEs,
                                                         weights->getShape()[1],
                                                         kNumMaccsPerPE);
    assert(bestWeightTilingDims != TilingDims::DimNH &&
           "Weights cannot be tiled by dimensions NH!");
    TilingDims bestOutputTilingDims =
            findBestTilingDims(outputs->getShape(), maxTileSize, 1, 1, kNumPEs);

    // Apply some constraints to simplify tiling logic.
    //
    // If weights = DimN or DimNC, then outputs must be DimNC, so that we will
    // copy out C channels of outputs after every tile.  In theory, we could
    // just keep more of the output pixels on the scratchpad and copy them only
    // when it's actually full but that's harder to manage (what if it fills up
    // in the middle of the next tile?).
    if (bestWeightTilingDims == DimN || bestWeightTilingDims == DimNC)
        bestOutputTilingDims = DimNC;

    // If inputs = DimNH, then outputs = DimNH. Strictly speaking this is not
    // necessarily required but it will greatly simplify memory management
    // (see above).
    if (bestInputTilingDims == DimNH)
        bestOutputTilingDims = DimNH;

    return { bestInputTilingDims, bestWeightTilingDims, bestOutputTilingDims };
}

// Determine the best basic tiling shape for this convolution layer.
//
// The algorithm first determines the dimensions along which the inputs,
// weights, and outputs will be tiled. Then based on those dimensions, we
// enumerate all possible basic tile shapes for inputs, weights, and outputs. A
// **basic** shape is the shape that all but potentially the last tile along a
// set of dimensions will use. This triplet of tile shapes defines a
// TilingConfig. The TilingConfig that maximizes the total combined size of
// input, weights, and output tiles is chosen as the best.
//
// To limit the number of possibilities, we only enumerate each dimension in
// certain increments. For example, input channels are only enumerated in
// multiples of kNumMaccsPerPE, and output channels are only enumerated in
// multiples in kNumPEs.
//
// This algorithm assumes that the maximum tile size for weights, inputs, and
// outputs are all the same and that they will reside in separate scratchpads
// (no sharing).
//
// Args:
//    op: The SMV convolution operator. All tensors must have been created with
//        createAllTensors() prior to calling this function.
// Returns:
//    The TilingConfig that describes the best tiling shapes.
//
TilingConfig TilingOptimizer::computeBasicTileShapes(SmvConvolutionOp* op) {
    Tensor* inputs = op->getInput(op->Inputs);
    Tensor* weights = op->getInput(op->Kernels);
    Tensor* outputs = op->getOutput(op->Outputs);
    int maxTileSize = SmvBackend::SpadSize() / inputs->getDataTypeSize();
    std::array<TilingDims, 3> strategies =
            determineBestTilingDims(inputs, weights, outputs, maxTileSize);
    TilingDims inputTilingDims = strategies[0];
    TilingDims weightTilingDims = strategies[1];
    TilingDims outputTilingDims = strategies[2];

    dout(2) << "Tiling dimensions chosen: \n"
            << "  input: " << inputTilingDims
            << ", weight: " << weightTilingDims
            << ", output: " << outputTilingDims << "\n";

    TensorShape inputsShape = inputs->getShape();
    TensorShape weightsShape = weights->getShape();
    TensorShape outputsShape = outputs->getShape();

    // There are four degrees of freedom we can play with in total:
    // N (batch), H (rows), C (channels), and P (ofmap).
    // Each tiling strategy may reduce this down to just three.
    // 1. Start with inputs. Enumerate all shapes that fit.
    // 2. Move on to weights. Enumerate all shapes that are compatible with
    //    the input shape and fit.
    // 3. Move on to outputs. If the weights don't need tiling, the outputs
    //    can be tiled independently; otherwise, based on the input and weights
    //    tile shapes, the output tile shape is completely determined.
    // For all tiling strategy, compute the total SRAM utilization. The highest
    // one is the chosen one.
    std::vector<TensorShape> inputConfigs;
    if (inputTilingDims == DimN) {
        for (int n = 1; n <= inputsShape[0]; n++) {
            TensorShape config;
            config = inputsShape;
            config[0] = n;
            if (config.storageSize() <= maxTileSize)
                inputConfigs.push_back(config);
            else
                break;
        }
    } else if (inputTilingDims == DimNC) {
        for (int n = 1; n <= inputsShape[0]; n++) {
            int minChannels = std::min(kNumMaccsPerPE, inputsShape[3]);
            for (int c = minChannels; c <= inputsShape[3]; c+=kNumMaccsPerPE) {
                TensorShape config;
                config = inputsShape;
                config[0] = n;
                config[3] = c;
                if (config.storageSize() <= maxTileSize)
                    inputConfigs.push_back(config);
                else
                    break;
            }
        }
    } else if (inputTilingDims == DimNH) {
        for (int n = 1; n <= inputsShape[0]; n++) {
            int minRows = std::min(inputsShape[1], weightsShape[1]);
            for (int r = minRows; r <= inputsShape[1]; r += 1) {
                TensorShape config = inputsShape;
                config[0] = n;
                config[1] = r;
                if (config.storageSize() <= maxTileSize)
                    inputConfigs.push_back(config);
                else
                    break;
            }
        }
    } else {
        inputConfigs.push_back(inputsShape);
    }
    assert(!inputConfigs.empty() && "No tiling configurations found!");

    // Fill in weights.
    std::list<TilingConfig> inputWeightConfigs;
    for (auto it = inputConfigs.begin(); it != inputConfigs.end(); ++it) {
        TensorShape& inputsShape = *it;
        if (weightTilingDims == DimN) {
            int minOfmaps = std::min(weightsShape[0], kNumPEs);
            for (int n = minOfmaps; n <= weightsShape[0]; n += kNumPEs) {
                TilingConfig config;
                config.weights = weightsShape;
                config.weights[0] = n;
                config.weights[3] = inputsShape[3];
                if (config.weights.storageSize() <= maxTileSize) {
                    config.inputs = inputsShape;
                    inputWeightConfigs.push_back(config);
                } else {
                    break;
                }
            }
        } else if (weightTilingDims == DimNC) {
            int minOfmaps = std::min(weightsShape[0], kNumPEs);
            int minChannels = std::min(weightsShape[3], kNumMaccsPerPE);
            for (int n = minOfmaps; n <= weightsShape[0]; n += kNumPEs) {
                TilingConfig config;
                config.weights = weightsShape;
                config.weights[0] = n;
                if (inputTilingDims == DimNC) {
                    // If the inputs are also tiled channelwise, then the
                    // weights have to take the same channel dimension.
                    config.weights[3] = inputsShape[3];
                    if (config.weights.storageSize() <= maxTileSize) {
                        config.inputs = inputsShape;
                        inputWeightConfigs.push_back(config);
                    } else {
                        break;
                    }
                } else {
                    // The weights can be independently tiled channelwise only
                    // if the inputs are not channelwise tiled.
                    for (int c = minChannels; c <= weightsShape[3];
                         c += kNumMaccsPerPE) {
                        config.weights[3] = c;
                        if (config.weights.storageSize() <= maxTileSize) {
                            config.inputs = inputsShape;
                            inputWeightConfigs.push_back(config);
                        } else {
                            break;
                        }
                    }
                }
            }
        } else {
            TilingConfig config;
            config.inputs = inputsShape;
            config.weights = weightsShape;
            inputWeightConfigs.push_back(config);
        }
    }
    assert(!inputWeightConfigs.empty() && "No tiling configurations found!");

    // Fill in outputs.
    std::vector<TilingConfig> fullConfigs;
    for (auto it = inputWeightConfigs.begin(); it != inputWeightConfigs.end();
         ++it) {
        int minChannels = std::min(it->weights[0], kNumPEs);
        bool weightsNeedTiling = (weightTilingDims != None);
        for (int c = minChannels; c <= weightsShape[0]; c += kNumPEs) {
            TilingConfig config = *it;
            config.outputs = outputsShape;
            config.outputs[0] = config.inputs[0];
            if (outputTilingDims == DimNH) {
                config.outputs[1] = op->computeOutputDim(config.inputs[1],
                                                         config.weights[1],
                                                         op->getRowStride(),
                                                         op->getPadding());
                config.outputs[3] = config.weights[0];
            } else {
                config.outputs[1] = outputsShape[1];
                config.outputs[3] = weightsNeedTiling ? config.weights[0] : c;
            }
            if (config.outputs.storageSize() <= maxTileSize) {
                fullConfigs.push_back(config);
            }
            // This means the output shape is uniquely determined, so we don't
            // need to explore any other output channel values.
            if (weightsNeedTiling)
                break;
        }
    }
    dout(2) << "Number of possible tiling configs: " << fullConfigs.size()
            << "\n";
    for (auto& config: fullConfigs) {
        dout(2) << "  inputs: " << config.inputs
                << ", weights: " << config.weights
                << ", outputs: " << config.outputs << "\n";
    }
    dout(2) << "==============\n";
    auto maxIt = std::max_element(
            fullConfigs.begin(),
            fullConfigs.end(),
            [](const TilingConfig& c1, const TilingConfig& c2) {
                return c1.getTotalSize() < c2.getTotalSize();
            });
    assert(maxIt != fullConfigs.end() && "Failed to get best tiling config!");
    // Fill in the tiling dims.
    (*maxIt).inputTilingDims = inputTilingDims;
    (*maxIt).weightTilingDims = weightTilingDims;
    (*maxIt).outputTilingDims = outputTilingDims;
    return *maxIt;
}

TiledTensor TilingOptimizer::generateTiledTensor(Tensor* tensor,
                                                 const TensorShape& tileShape,
                                                 std::vector<int> halos,
                                                 Workspace* workspace) {
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
    for (auto tileIndex = tiledTensor.startIndex(); !tileIndex.end();
         ++tileIndex) {
        std::vector<int> currentTileShape(ndims);
        for (int i = 0; i < ndims; i++) {
            currentTileShape[i] =
                    std::min(inputShape[i] - currentOrigin[i], tileShape[i]);
        }
        TensorShape currentShape(
                currentTileShape, tileShape.getLayout(), SmvBackend::Alignment);
        std::string tileName =
                tensor->getName() + "/tile:" + std::to_string((int)tileIndex);
        Tensor* tile = new Tensor(tileName, currentShape);
        tile->allocateStorage(tensor->getDataType());
        copyTensorRegion(tile,
                         tensor,
                         { 0, 0, 0, 0 },
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
    workspace->addTiledTensor(tiledTensor);
    return tiledTensor;
}

TiledTensor TilingOptimizer::generateDimNHOutputTiledTensor(
        SmvConvolutionOp* op,
        const TiledTensor& inputTiledTensor,
        const TiledTensor& weightsTiledTensor,
        const TensorShape& maxOutputTileSize,
        Tensor* outputTensor,
        bool copyData) {
    const TensorShape& inputShape = inputTiledTensor.getShape();
    const TensorShape& weightsShape = weightsTiledTensor.getShape();
    const TensorShape& outputShape = outputTensor->getShape();
    std::vector<int> numBlocksInDim{ inputShape[0], inputShape[1],
                                     inputShape[2], weightsShape[0] };
    TiledTensor outputTiledTensor(
            TensorShape(numBlocksInDim, inputShape.getLayout()));
    const int ndims = outputShape.ndims();
    std::vector<int> currentOrigin(ndims, 0);
    auto inputIndex = inputTiledTensor.startIndex();
    auto weightIndex = weightsTiledTensor.startIndex();
    auto outputIndex = outputTiledTensor.startIndex();
    int weightRows = op->getWeightRows();
    // For even-sized filtered, FRAC_CEIL is needed to correctly handle padding.
    int topRowPad = (op->getPadding() == SamePadding)
                            ? FRAC_CEIL(weightRows - 1, 2)
                            : 0;
    int bottomRowPad = (op->getPadding() == SamePadding)
                               ? (weightRows - 1 - topRowPad)
                               : 0;
    for (int n = 0; n < numBlocksInDim[0]; n++) {
        for (int h = 0; h < numBlocksInDim[1]; h++) {
            for (int w = 0; w < numBlocksInDim[2]; w++) {
                for (int c = 0; c < numBlocksInDim[3]; c++) {
                    const Tensor* inputTile =
                            inputTiledTensor[inputIndex(n, h, w, 0)];
                    const Tensor* weightsTile =
                            weightsTiledTensor[weightIndex(c, 0, 0, 0)];
                    const TensorShape& inputTileShape = inputTile->getShape();

                    // DimNH tiling only affects rows, not columns.
                    int effInputRows = inputTileShape[1];
                    if (h == 0)
                        effInputRows += topRowPad;
                    else if (h == numBlocksInDim[1] - 1)
                        effInputRows += bottomRowPad;
                    int outputRows = op->computeOutputDim(effInputRows,
                                                          weightRows,
                                                          op->getRowStride(),
                                                          ValidPadding);
                    TensorShape outputTileShape(
                            { inputTileShape[0], outputRows, inputTileShape[2],
                              weightsTile->getShape()[0] },
                            outputTensor->getShape().getLayout(),
                            SmvBackend::Alignment);
                    assert(outputTileShape.storageSize() <=
                                   maxOutputTileSize.storageSize() &&
                           "DimNH input tiling results in output tile sizes "
                           "larger than the max tile size!");
                    int oi = outputIndex(n, h, w, c);
                    std::string tileName = outputTensor->getName() +
                                           "/tile:" + std::to_string((int)oi);
                    Tensor* outputTile = new Tensor(tileName, outputTileShape);
                    outputTile->allocateStorage(outputTensor->getDataType());
                    if (copyData) {
                        copyTensorRegion(outputTile,
                                         outputTensor,
                                         { 0, 0, 0, 0 },
                                         currentOrigin,
                                         outputTileShape.dims());
                        for (int i = ndims - 1; i >= 0; i--) {
                            currentOrigin[i] += outputTileShape[i];
                            if (currentOrigin[i] >= outputShape[i])
                                currentOrigin[i] = 0;
                            else
                                break;
                        }
                    }
                    outputTiledTensor[oi] = outputTile;
                }
            }
        }
    }
    op->getWorkspace()->addTiledTensor(outputTiledTensor);
    return outputTiledTensor;
}

std::array<TiledTensor, 3> TilingOptimizer::doTiling(SmvConvolutionOp* op) {
    auto input = op->getInput(SmvConvolutionOp::Inputs);
    auto kernels = op->getInput(SmvConvolutionOp::Kernels);
    auto output = op->getOutput(SmvConvolutionOp::Outputs);
    TilingConfig tileConfig = TilingOptimizer::computeBasicTileShapes(op);
    std::vector<int> inputHalos{ 0, op->getWeightRows() - op->getRowStride(),
                                 op->getWeightCols() - op->getColStride(), 0 };
    TiledTensor tiledInputs = TilingOptimizer::generateTiledTensor(
            input, tileConfig.inputs, inputHalos, op->getWorkspace());
    TiledTensor tiledWeights = TilingOptimizer::generateTiledTensor(
            kernels, tileConfig.weights, { 0, 0, 0, 0 }, op->getWorkspace());
    TiledTensor tiledOutputs;
    if (tileConfig.outputTilingDims == DimNH) {
        tiledOutputs = TilingOptimizer::generateDimNHOutputTiledTensor(
                op,
                tiledInputs,
                tiledWeights,
                tileConfig.outputs,
                output,
                true);
    } else {
        tiledOutputs = TilingOptimizer::generateTiledTensor(
                output, tileConfig.outputs, { 0, 0, 0, 0 }, op->getWorkspace());
    }
    return { tiledInputs, tiledWeights, tiledOutputs };
}

}  // namespace conv
}  // namespace smv
}  // namespace smaug
