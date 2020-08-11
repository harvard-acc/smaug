#include <algorithm>

#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/smv/smv_convolution_op.h"
#include "smaug/operators/smv/smv_convolution_tiling.h"
#include "smaug/utility/debug_stream.h"

namespace smaug {
namespace smv {
namespace conv {

std::array<TilingDims, 3> TilingOptimizer::determineBestTilingDims(
        Tensor* inputs, Tensor* weights, Tensor* outputs, int maxTileSize) {
    // Determine the best tiling strategy for each of inputs, weights, and
    // outputs. Don't try to figure out the actual tile sizes yet.
    TilingDims bestInputTilingDims =
            findBestTilingDims(inputs->getShape(),
                               maxTileSize,
                               { 1, weights->getShape()[1],
                                 inputs->getShape()[2], kNumMaccsPerPE });
    TilingDims bestWeightTilingDims =
            findBestTilingDims(weights->getShape(),
                               maxTileSize,
                               { kNumPEs, weights->getShape()[1],
                                 weights->getShape()[2], kNumMaccsPerPE });
    assert(bestWeightTilingDims != TilingDims::DimNH &&
           "Weights cannot be tiled by dimensions NH!");
    TilingDims bestOutputTilingDims =
            findBestTilingDims(outputs->getShape(),
                               maxTileSize,
                               { 1, 1, outputs->getShape()[2], kNumPEs });

    // Apply some constraints to simplify tiling logic.
    //
    // If weights = DimN or DimNC, then outputs must be DimNC, so that we will
    // copy out C channels of outputs after every tile.  In theory, we could
    // just keep more of the output pixels on the scratchpad and copy them only
    // when it's actually full but that's harder to manage (what if it fills up
    // in the middle of the next tile?).
    if (needsNwiseTiling(bestWeightTilingDims))
        bestOutputTilingDims = DimNC;

    // If inputs require rowwise tiling, then outputs also require rowwise
    // tiling. Strictly speaking this is not necessarily required but it will
    // greatly simplify memory management (see above).
    if (needsHwiseTiling(bestInputTilingDims)) {
        if (needsCwiseTiling(bestOutputTilingDims))
            bestOutputTilingDims = DimNCH;
        else
            bestOutputTilingDims = DimNH;
    }

    return { bestInputTilingDims, bestWeightTilingDims, bestOutputTilingDims };
}

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

    dout(2) << "  Tiling dimensions chosen:\n"
            << "    input: " << inputTilingDims
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
        std::vector<int> minShape = inputsShape.dims();
        minShape[0] = 1;
        enum4DTensorTilingConfigs(inputsShape,
                                  maxTileSize,
                                  minShape,
                                  { 1, 1, 1, 1 },
                                  inputConfigs);
    } else if (inputTilingDims == DimNC) {
        std::vector<int> minShape = inputsShape.dims();
        minShape[0] = 1;
        minShape[3] = kNumMaccsPerPE;
        enum4DTensorTilingConfigs(inputsShape,
                                  maxTileSize,
                                  minShape,
                                  { 1, 1, 1, kNumMaccsPerPE },
                                  inputConfigs);
    } else if (inputTilingDims == DimNH) {
        std::vector<int> minShape = inputsShape.dims();
        minShape[0] = 1;
        minShape[1] = weightsShape[1];
        enum4DTensorTilingConfigs(inputsShape,
                                  maxTileSize,
                                  minShape,
                                  { 1, op->getRowStride(), 1, 1 },
                                  inputConfigs);
    } else if (inputTilingDims == DimNCH) {
        std::vector<int> minShape = { 1, weightsShape[1], inputsShape[2],
                                      kNumMaccsPerPE };
        std::vector<int> strides = { 1, op->getRowStride(), 1, kNumMaccsPerPE };
        enum4DTensorTilingConfigs(
                inputsShape, maxTileSize, minShape, strides, inputConfigs);
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
                if (needsCwiseTiling(inputTilingDims)) {
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
        } else if (weightTilingDims == DimNH || weightTilingDims == DimNCH) {
            assert(false && "Weights can't be tiled rowwise!");
        } else {
            TilingConfig config;
            config.inputs = inputsShape;
            config.weights = weightsShape;
            if (needsCwiseTiling(inputTilingDims)) {
                // This can happen with small weights. If the inputs are tiled
                // channelwise, then the weight tile need to have the same
                // number of channels.
                config.weights[3] = inputsShape[3];
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
        for (int c = minChannels; c <= weightsShape[0]; c += kNumPEs) {
            TilingConfig config = *it;
            config.outputs = outputsShape;
            config.outputs[0] = config.inputs[0];
            if (needsHwiseTiling(outputTilingDims)) {
                int padding = op->getPadding() == SamePadding
                                      ? FRAC_CEIL(config.weights[1] - 1, 2)
                                      : 0;
                config.outputs[1] = op->computeOutputDim(config.inputs[1],
                                                         config.weights[1],
                                                         op->getRowStride(),
                                                         padding);
                config.outputs[3] = config.weights[0];
            } else {
                config.outputs[1] = outputsShape[1];
                if (weightsNeedTiling)
                    config.outputs[3] = config.weights[0];
                // If the weights don't need tiling and the outputs need tiling,
                // the channel size of the output tile size can be determined
                // independently.
                else if (outputTilingDims != None)
                    config.outputs[3] = c;
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
    (*maxIt).inputTilingDims = inputTilingDims;
    (*maxIt).weightTilingDims = weightTilingDims;
    (*maxIt).outputTilingDims = outputTilingDims;
    return *maxIt;
}

TiledTensor TilingOptimizer::generateRowwiseOutputTiledTensor(
        SmvConvolutionOp* op,
        const TiledTensor& inputTiledTensor,
        const TiledTensor& weightsTiledTensor,
        const TensorShape& maxOutputTileSize,
        Tensor* outputTensor,
        bool copyData) {
    const TensorShape& inputShape = inputTiledTensor.getShape();
    const TensorShape& weightsShape = weightsTiledTensor.getShape();
    const TensorShape& outputShape = outputTensor->getShape();
    int weightRows = op->getWeightRows();
    int weightCols = op->getWeightCols();
    bool samePadding = op->getPadding() == SamePadding;
    // For even-sized filtered, FRAC_CEIL is needed to correctly handle padding.
    std::vector<int> inputPadding = op->getInputPadding();
    int topRowPad = inputPadding[0];
    int bottomRowPad = inputPadding[1];
    int leftColPad = inputPadding[2];
    int rightColPad = inputPadding[3];
    std::vector<int> numBlocksInDim{ inputShape[0], inputShape[1],
                                     inputShape[2], weightsShape[0] };
    // Due to stride > 1, there is a case where the last rowwise tile doesn't
    // have enough rows for convolution. If so, we need to decrease the row
    // dimension by 1 in the output tiled tensor.
    int lastTileRows =
            inputTiledTensor[inputTiledTensor.size() - 1]->getShape()[1];
    if (lastTileRows + bottomRowPad < weightRows)
        numBlocksInDim[1]--;
    TiledTensor outputTiledTensor(
            TensorShape(numBlocksInDim, inputShape.getLayout()), outputTensor);
    const int ndims = outputShape.ndims();
    std::vector<int> currentOrigin(ndims, 0);
    auto inputIndex = inputTiledTensor.startIndex();
    auto weightIndex = weightsTiledTensor.startIndex();
    auto outputIndex = outputTiledTensor.startIndex();
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
                    int effInputCols =
                            inputTileShape[2] + leftColPad + rightColPad;
                    int outputRows = op->computeOutputDim(effInputRows,
                                                          weightRows,
                                                          op->getRowStride(),
                                                          ValidPadding);
                    int outputCols = op->computeOutputDim(effInputCols,
                                                          weightCols,
                                                          op->getColStride(),
                                                          ValidPadding);
                    TensorShape outputTileShape(
                            { inputTileShape[0], outputRows, outputCols,
                              weightsTile->getShape()[0] },
                            outputTensor->getShape().getLayout(),
                            SmvBackend::Alignment);
                    assert(outputTileShape.storageSize() <=
                                   maxOutputTileSize.storageSize() &&
                           "DimNH input tiling results in output tile sizes "
                           "larger than the max tile size!");
                    int oi = outputIndex(n, h, w, c);
                    std::string tileName = op->getName() + ":" +
                                           outputTensor->getName() +
                                           "/tile:" + std::to_string((int)oi);
                    Tensor* outputTile = new Tensor(tileName, outputTileShape);
                    outputTile->allocateStorage(outputTensor->getDataType());
                    outputTiledTensor.setTile(
                            oi, currentOrigin, outputTile, copyData);
                    for (int i = ndims - 1; i >= 0; i--) {
                        currentOrigin[i] += outputTileShape[i];
                        if (currentOrigin[i] >= outputShape[i])
                            currentOrigin[i] = 0;
                        else
                            break;
                    }
                }
            }
        }
    }
    op->getWorkspace()->addTiledTensor(outputTiledTensor);
    dout(1) << "  Tiled Tensor " << outputTensor->getName() << "(rowwise):\n"
            << "    original tensor shape: " << outputTensor->getShape() << "\n"
            << "    number of tiles: " << outputTiledTensor.size() << "\n";
    return outputTiledTensor;
}

std::array<TiledTensor, 3> TilingOptimizer::doTiling(SmvConvolutionOp* op) {
    auto input = op->getInput(SmvConvolutionOp::Inputs);
    auto kernels = op->getInput(SmvConvolutionOp::Kernels);
    auto output = op->getOutput(SmvConvolutionOp::Outputs);
    TilingConfig tileConfig = TilingOptimizer::computeBasicTileShapes(op);
    TiledTensor tiledInputs = generateTiledTensor(input,
                                                  tileConfig.inputs,
                                                  op,
                                                  op->getWeightRows(),
                                                  op->getWeightCols(),
                                                  op->getRowStride(),
                                                  op->getColStride(),
                                                  op->getPadding());
    // Copy data for the weight tiles since the data is read-only.
    TiledTensor tiledWeights = generateTiledTensorAndCopyData(
            kernels, tileConfig.weights, op);
    TiledTensor tiledOutputs;
    if (needsHwiseTiling(tileConfig.outputTilingDims)) {
        tiledOutputs = TilingOptimizer::generateRowwiseOutputTiledTensor(
                op,
                tiledInputs,
                tiledWeights,
                tileConfig.outputs,
                output,
                false);
    } else {
        tiledOutputs = generateTiledTensor(output, tileConfig.outputs, op);
    }
    return { tiledInputs, tiledWeights, tiledOutputs };
}

}  // namespace conv
}  // namespace smv
}  // namespace smaug
