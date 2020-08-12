#include <algorithm>

#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/smv/smv_batch_norm_op.h"
#include "smaug/operators/smv/smv_batch_norm_tiling.h"
#include "smaug/utility/debug_stream.h"

namespace smaug {
namespace smv {
namespace bn {

std::array<TilingDims, 2> TilingOptimizer::determineBestTilingDims(
        Tensor* inputs, Tensor* weights, int maxTileSize) {
    // Determine the best tiling dims for each of inputs and weights. The
    // outputs have the same shape as the inputs and should use the same tiling
    // dims.
    const TensorShape& inputShape = inputs->getShape();
    TilingDims bestInputTilingDims;
    if (inputShape.ndims() == 4) {
        bestInputTilingDims = findBestTilingDims(
                inputShape,
                maxTileSize,
                { 1, kVectorSize, kVectorSize, kVectorSize });
    } else {
        bestInputTilingDims =
                findBestTilingDims(inputShape, maxTileSize, { 1, kVectorSize });
    }
    TilingDims bestWeightTilingDims = findBestTilingDims(
            weights->getShape(), maxTileSize, { 4, kVectorSize });

    return { bestInputTilingDims, bestWeightTilingDims };
}

void TilingOptimizer::enumPostFCTilingConfigs(
        TensorShape inputsShape,
        TensorShape weightsShape,
        int maxTileSize,
        std::array<TilingDims, 2> strategies,
        std::list<TilingConfig>& fullConfigs) {
    TilingDims inputTilingDims = strategies[0];
    TilingDims weightTilingDims = strategies[1];
    // Supported tiling dims: None, DimN and DimNC for inputs. None and DimNC
    // for weights.
    // The tiling config enumeration goes as follows:
    // 1. Start with inputs. Enumerate all shapes that fit.
    // 2. Move on to weights and outputs. Enumerate all shapes that are
    //    compatible with the input shape and fit. The outputs tile use the same
    //    tile shape as the inputs.
    // 3. For all tiling strategy, compute the total SRAM utilization. The
    //    highest one is the chosen one.
    assert(inputTilingDims == None || inputTilingDims == DimN ||
           inputTilingDims == DimNC);
    assert(weightTilingDims == None || weightTilingDims == DimNC);
    std::vector<TensorShape> inputsConfigs;
    if (inputTilingDims == DimN) {
        std::vector<int> minShape = inputsShape.dims();
        minShape[0] = 1;
        enum2DTensorTilingConfigs(
                inputsShape, maxTileSize, minShape, { 1, 1 }, inputsConfigs);
    } else if (inputTilingDims == DimNC) {
        enum2DTensorTilingConfigs(inputsShape,
                                  maxTileSize,
                                  { 1, kVectorSize },
                                  { 1, kVectorSize },
                                  inputsConfigs);
    } else {
        inputsConfigs.push_back(inputsShape);
    }
    assert(!inputsConfigs.empty() && "No tiling configurations found!");

    // Fill in weights and outputs.
    for (auto it = inputsConfigs.begin(); it != inputsConfigs.end(); ++it) {
        TensorShape& inputsConfig = *it;
        if (weightTilingDims == DimNC) {
            if (needsCwiseTiling(inputTilingDims)) {
                // If the inputs are also tiled activation-wise, then the
                // weights have to take the same activations dimension.
                TilingConfig config;
                config.weights = weightsShape;
                config.weights[1] = inputsConfig[1];
                if (config.weights.storageSize() <= maxTileSize) {
                    config.inputs = inputsConfig;
                    config.outputs = inputsConfig;
                    fullConfigs.push_back(config);
                } else {
                    break;
                }
            } else {
                int minChannels = std::min(weightsShape[1], kVectorSize);
                for (int c = minChannels; c <= weightsShape[1];
                     c += kVectorSize) {
                    TilingConfig config;
                    config.weights = weightsShape;
                    config.weights[1] = c;
                    if (config.weights.storageSize() <= maxTileSize) {
                        config.inputs = inputsConfig;
                        config.outputs = inputsConfig;
                        fullConfigs.push_back(config);
                    } else {
                        break;
                    }
                }
            }
        } else {
            TilingConfig config(inputsConfig, weightsShape, inputsConfig);
            fullConfigs.push_back(config);
        }
    }
    assert(!fullConfigs.empty() && "No tiling configurations found!");
}

void TilingOptimizer::enumPostConvTilingConfigs(
        TensorShape inputsShape,
        TensorShape weightsShape,
        int maxTileSize,
        std::array<TilingDims, 2> strategies,
        std::list<TilingConfig>& fullConfigs) {
    TilingDims inputTilingDims = strategies[0];
    TilingDims weightTilingDims = strategies[1];
    // Supported tiling dims: DimN, DimNC, DimNH, DimNW, DimNHW, DimNCH and
    // DimNCW for inputs. None for weights for now. For a 32KB weights spad, it
    // would mean the weights have more than 4096 channels if tiling is
    // required.
    // TODO: add other tiling dims for weights if we need that later.
    // Enumerate all input shapes that fit and then fill the
    // tiling configurations with weights and outputs. For all tiling strategy,
    // compute the total SRAM utilization. The highest one is the chosen one.
    assert(inputTilingDims == None || inputTilingDims == DimN ||
           inputTilingDims == DimNC || inputTilingDims == DimNH ||
           inputTilingDims == DimNW || inputTilingDims == DimNHW ||
           inputTilingDims == DimNCH || inputTilingDims == DimNCW);
    assert(weightTilingDims == None);
    std::vector<TensorShape> inputsConfigs;
    if (inputTilingDims == DimN) {
        std::vector<int> minShape = inputsShape.dims();
        minShape[0] = 1;
        enum4DTensorTilingConfigs(inputsShape,
                                  maxTileSize,
                                  minShape,
                                  { 1, 1, 1, 1 },
                                  inputsConfigs);
    } else if (inputTilingDims == DimNC) {
        std::vector<int> minShape = inputsShape.dims();
        minShape[0] = 1;
        minShape[3] = kVectorSize;
        enum4DTensorTilingConfigs(inputsShape,
                                  maxTileSize,
                                  minShape,
                                  { 1, 1, 1, kVectorSize },
                                  inputsConfigs);
    } else if (inputTilingDims == DimNH) {
        std::vector<int> minShape = inputsShape.dims();
        minShape[0] = 1;
        minShape[1] = kVectorSize;
        enum4DTensorTilingConfigs(inputsShape,
                                  maxTileSize,
                                  minShape,
                                  { 1, kVectorSize, 1, 1 },
                                  inputsConfigs);
    } else if (inputTilingDims == DimNW) {
        std::vector<int> minShape = inputsShape.dims();
        minShape[0] = 1;
        minShape[2] = kVectorSize;
        enum4DTensorTilingConfigs(inputsShape,
                                  maxTileSize,
                                  minShape,
                                  { 1, 1, kVectorSize, 1 },
                                  inputsConfigs);
    } else if (inputTilingDims == DimNHW) {
        std::vector<int> minShape = { 1, kVectorSize, kVectorSize,
                                      inputsShape[3] };
        enum4DTensorTilingConfigs(inputsShape,
                                  maxTileSize,
                                  minShape,
                                  { 1, kVectorSize, kVectorSize, 1 },
                                  inputsConfigs);
    } else if (inputTilingDims == DimNCH) {
        std::vector<int> minShape = { 1, kVectorSize, inputsShape[2],
                                      kVectorSize };
        enum4DTensorTilingConfigs(inputsShape,
                                  maxTileSize,
                                  minShape,
                                  { 1, kVectorSize, 1, kVectorSize },
                                  inputsConfigs);
    } else if (inputTilingDims == DimNCW) {
        std::vector<int> minShape = { 1, inputsShape[1], kVectorSize,
                                      kVectorSize };
        enum4DTensorTilingConfigs(inputsShape,
                                  maxTileSize,
                                  minShape,
                                  { 1, 1, kVectorSize, kVectorSize },
                                  inputsConfigs);
    } else {
        inputsConfigs.push_back(inputsShape);
    }
    assert(!inputsConfigs.empty() && "No tiling configurations found!");

    // Fill in weights and outputs.
    for (auto it = inputsConfigs.begin(); it != inputsConfigs.end(); ++it) {
        TilingConfig config(*it, weightsShape, *it);
        fullConfigs.push_back(config);
    }
    assert(!fullConfigs.empty() && "No tiling configurations found!");
}

TilingConfig TilingOptimizer::computeBasicTileShapes(Tensor* inputs,
                                                     Tensor* weights,
                                                     Tensor* outputs) {
    int maxTileSize = SmvBackend::SpadSize() / inputs->getDataTypeSize();
    // The outputs have the same shape as the inputs. No need to tile it.
    assert(inputs->getShape() == outputs->getShape());
    std::array<TilingDims, 2> strategies =
            determineBestTilingDims(inputs, weights, maxTileSize);
    TilingDims inputTilingDims = strategies[0];
    TilingDims weightTilingDims = strategies[1];
    TilingDims outputTilingDims = inputTilingDims;

    dout(2) << "  Tiling dimensions chosen: \n"
            << "    input: " << inputTilingDims
            << ", weight: " << weightTilingDims
            << ", output: " << inputTilingDims << "\n";

    TensorShape inputsShape = inputs->getShape();
    TensorShape weightsShape = weights->getShape();
    std::list<TilingConfig> fullConfigs;
    bool isPostConv = (inputs->ndims() == 4);
    if (isPostConv) {
        enumPostConvTilingConfigs(inputsShape,
                                  weightsShape,
                                  maxTileSize,
                                  strategies,
                                  fullConfigs);
    } else {
        enumPostFCTilingConfigs(inputsShape,
                                weightsShape,
                                maxTileSize,
                                strategies,
                                fullConfigs);
    }

    dout(2) << "  Number of possible tiling configs: " << fullConfigs.size()
            << "\n";
    for (auto& config : fullConfigs)
        dout(2) << "    " << config << "\n";
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

std::array<TiledTensor, 3> TilingOptimizer::doTiling(SmvBatchNormOp* op) {
    auto inputs = op->getInput(SmvBatchNormOp::Inputs);
    auto mean = op->getInput(SmvBatchNormOp::Mean);
    auto variance = op->getInput(SmvBatchNormOp::Variance);
    auto gamma = op->getInput(SmvBatchNormOp::Gamma);
    auto beta = op->getInput(SmvBatchNormOp::Beta);
    // Concatenate the four weight tensors into one.
    auto weights = concatTensors(
            { mean, variance, gamma, beta }, 0, op->getWorkspace());
    auto outputs = op->getOutput(SmvBatchNormOp::Outputs);
    TilingConfig tileConfig =
            TilingOptimizer::computeBasicTileShapes(inputs, weights, outputs);
    TiledTensor tiledInputs =
            generateTiledTensor(inputs, tileConfig.inputs, op);
    // Copy data for the weight tiles since the data is read-only.
    TiledTensor tiledWeights =
            generateTiledTensorAndCopyData(weights, tileConfig.weights, op);
    TiledTensor tiledOutputs =
            generateTiledTensor(outputs, tileConfig.inputs, op);
    return { tiledInputs, tiledWeights, tiledOutputs };
}

}  // namespace bn
}  // namespace smv
}  // namespace smaug
