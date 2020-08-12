#include <algorithm>

#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/smv/smv_inner_product_op.h"
#include "smaug/operators/smv/smv_inner_product_tiling.h"
#include "smaug/utility/debug_stream.h"

namespace smaug {
namespace smv {
namespace fc {

std::array<TilingDims, 3> TilingOptimizer::determineBestTilingDims(
        Tensor* inputs, Tensor* weights, Tensor* outputs, int maxTileSize) {
    // Determine the best tiling strategy for each of inputs, weights, and
    // outputs. Don't try to figure out the actual tile sizes yet.
    TilingDims bestInputTilingDims = findBestTilingDims(
            inputs->getShape(), maxTileSize, { 1, kNumMaccsPerPE });
    TilingDims bestWeightTilingDims = findBestTilingDims(
            weights->getShape(), maxTileSize, { kNumPEs, kNumMaccsPerPE });
    TilingDims bestOutputTilingDims = findBestTilingDims(
            outputs->getShape(), maxTileSize, { 1, kNumPEs });

    // Apply some constraints to simplify tiling logic.
    //
    // If weights require tiling on neurons, then outputs must be DimNC (if
    // outputs require tiling), so that we will copy out C neurons of outputs
    // after every tile.
    if (needsNwiseTiling(bestWeightTilingDims) && bestOutputTilingDims != None)
        bestOutputTilingDims = DimNC;

    return { bestInputTilingDims, bestWeightTilingDims, bestOutputTilingDims };
}

TilingConfig TilingOptimizer::computeBasicTileShapes(SmvInnerProductOp* op) {
    Tensor* inputs = op->getInput(op->Inputs);
    Tensor* weights = op->getInput(op->Weights);
    Tensor* outputs = op->getOutput(op->Outputs);
    int maxTileSize = SmvBackend::SpadSize() / inputs->getDataTypeSize();
    std::array<TilingDims, 3> strategies =
            determineBestTilingDims(inputs, weights, outputs, maxTileSize);
    TilingDims inputTilingDims = strategies[0];
    TilingDims weightTilingDims = strategies[1];
    TilingDims outputTilingDims = strategies[2];

    dout(1) << "  Tiling dimensions chosen:\n"
            << "    input: " << inputTilingDims
            << ", weight: " << weightTilingDims
            << ", output: " << outputTilingDims << "\n";

    TensorShape inputsShape = inputs->getShape();
    TensorShape weightsShape = weights->getShape();
    TensorShape outputsShape = outputs->getShape();

    // There are two degrees of freedom we can play with in total:
    // N (batch/neuron), C (activation).
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
        enum2DTensorTilingConfigs(inputsShape,
                                  maxTileSize,
                                  { 1, inputsShape[1] },
                                  { 1, 1 },
                                  inputConfigs);
    } else if (inputTilingDims == DimNC) {
        std::vector<int> minShape = inputsShape.dims();
        enum2DTensorTilingConfigs(inputsShape,
                                  maxTileSize,
                                  { 1, kNumMaccsPerPE },
                                  { 1, kNumMaccsPerPE },
                                  inputConfigs);

    } else {
        inputConfigs.push_back(inputsShape);
    }
    assert(!inputConfigs.empty() && "No tiling configurations found!");

    // Fill in weights.
    std::list<TilingConfig> inputWeightConfigs;
    for (auto it = inputConfigs.begin(); it != inputConfigs.end(); ++it) {
        const TensorShape& inputsShape = *it;
        if (weightTilingDims == DimN) {
            int minOfmaps = std::min(weightsShape[0], kNumPEs);
            for (int n = minOfmaps; n <= weightsShape[0]; n += kNumPEs) {
                TilingConfig config;
                config.weights = TensorShape({ n, inputsShape[1] },
                                             inputsShape.getLayout(),
                                             SmvBackend::Alignment);
                if (config.weights.storageSize() <= maxTileSize) {
                    config.inputs = inputsShape;
                    inputWeightConfigs.push_back(config);
                } else {
                    break;
                }
            }
        } else if (weightTilingDims == DimNC) {
            int minNeurons = std::min(weightsShape[0], kNumPEs);
            int minActs = std::min(weightsShape[1], kNumMaccsPerPE);
            for (int n = minNeurons; n <= weightsShape[0]; n += kNumPEs) {
                TilingConfig config;
                config.weights = weightsShape;
                config.weights[0] = n;
                if (needsCwiseTiling(inputTilingDims)) {
                    // If the inputs are also tiled activation-wise, then the
                    // weights have to take the same activations dimension.
                    config.weights[1] = inputsShape[1];
                    if (config.weights.storageSize() <= maxTileSize) {
                        config.inputs = inputsShape;
                        inputWeightConfigs.push_back(config);
                    } else {
                        break;
                    }
                } else {
                    // The weights can be independently tiled activation-wise
                    // only if the inputs are not tiled on activations.
                    for (int c = minActs; c <= weightsShape[1];
                         c += kNumMaccsPerPE) {
                        config.weights[1] = c;
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
            if (needsCwiseTiling(inputTilingDims)) {
                // This can happen with small weights. If the inputs are tiled
                // channelwise, then the weight tile need to have the same
                // number of channels.
                config.weights[1] = inputsShape[1];
            }
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
        bool outputsNeedTiling = (outputTilingDims != None);
        for (int c = minChannels; c <= weightsShape[0]; c += kNumPEs) {
            TilingConfig config = *it;
            config.outputs = outputsShape;
            config.outputs[0] = config.inputs[0];
            if (weightsNeedTiling && outputsNeedTiling) {
                config.outputs[1] = config.weights[0];
            } else if (outputsNeedTiling) {
                // This could rarely happen, but for completeness let's keep it.
                // If the weights don't need tiling and the outputs need tiling,
                // the channel size of the output tile size can be determined
                // independently.
                config.outputs[1] = c;
            }
            if (config.outputs.storageSize() <= maxTileSize) {
                fullConfigs.push_back(config);
            }
            // This means the output shape is uniquely determined, so we don't
            // need to explore any other output channel values.
            if (weightsNeedTiling || outputTilingDims == None)
                break;
        }
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
    maxIt->inputTilingDims = inputTilingDims;
    maxIt->weightTilingDims = weightTilingDims;
    maxIt->outputTilingDims = outputTilingDims;
    return *maxIt;
}

std::array<TiledTensor, 3> TilingOptimizer::doTiling(SmvInnerProductOp* op) {
    auto input = op->getInput(SmvInnerProductOp::Inputs);
    auto kernels = op->getInput(SmvInnerProductOp::Weights);
    auto output = op->getOutput(SmvInnerProductOp::Outputs);
    TilingConfig tileConfig = TilingOptimizer::computeBasicTileShapes(op);
    TiledTensor tiledInputs = generateTiledTensor(input, tileConfig.inputs, op);
    // Copy data for the weight tiles since the data is read-only.
    TiledTensor tiledWeights =
            generateTiledTensorAndCopyData(kernels, tileConfig.weights, op);
    TiledTensor tiledOutputs =
            generateTiledTensor(output, tileConfig.outputs, op);
    return { tiledInputs, tiledWeights, tiledOutputs };
}

}  // namespace fc
}  // namespace smv
}  // namespace smaug
