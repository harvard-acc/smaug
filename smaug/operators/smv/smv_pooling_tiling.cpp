#include <algorithm>

#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/smv/smv_pooling_op.h"
#include "smaug/operators/smv/smv_pooling_tiling.h"
#include "smaug/utility/debug_stream.h"

namespace smaug {
namespace smv {
namespace pool {

std::array<TilingDims, 2> TilingOptimizer::determineBestTilingDims(
        Tensor* inputs,
        Tensor* outputs,
        int maxTileSize,
        std::pair<int, int> poolSize) {
    // Determine the best tiling strategy for each of inputs and outputs. Don't
    // try to figure out the actual tile sizes yet.
    TilingDims bestInputTilingDims = findBestTilingDims(
            inputs->getShape(),
            maxTileSize,
            { 1, poolSize.first, poolSize.second, kVectorSize });
    TilingDims bestOutputTilingDims = findBestTilingDims(
            outputs->getShape(), maxTileSize, { 1, 1, 1, kVectorSize });

    // Apply some constraints to simplify tiling logic.
    //
    // If inputs require rowwise/columnwise tiling, then outputs also require
    // rowwise/columnwise tiling. Strictly speaking this is not necessarily
    // required but it will greatly simplify memory management.
    if (needsHwiseTiling(bestInputTilingDims)) {
        if (needsCwiseTiling(bestOutputTilingDims))
            bestOutputTilingDims = DimNCH;
        else if (needsWwiseTiling(bestOutputTilingDims))
            bestOutputTilingDims = DimNHW;
        else
            bestOutputTilingDims = DimNH;
    }
    if (needsWwiseTiling(bestInputTilingDims)) {
        if (needsCwiseTiling(bestOutputTilingDims))
            bestOutputTilingDims = DimNCW;
        else if (needsHwiseTiling(bestOutputTilingDims))
            bestOutputTilingDims = DimNHW;
        else
            bestOutputTilingDims = DimNW;
    }

    return { bestInputTilingDims, bestOutputTilingDims };
}

TilingConfig TilingOptimizer::computeBasicTileShapes(SmvPoolingOp* op) {
    Tensor* inputs = op->getInput(op->Inputs);
    Tensor* outputs = op->getOutput(op->Outputs);
    int maxTileSize = SmvBackend::SpadSize() / inputs->getDataTypeSize();
    std::pair<int, int> poolSize = op->getPoolingSize();
    std::pair<int, int> poolStride = op->getPoolingStride();
    std::array<TilingDims, 2> strategies =
            determineBestTilingDims(inputs, outputs, maxTileSize, poolSize);
    TilingDims inputTilingDims = strategies[0];
    TilingDims outputTilingDims = strategies[1];

    dout(2) << "  Tiling dimensions chosen: \n"
            << "    input: " << inputTilingDims
            << ", output: " << outputTilingDims << "\n";

    TensorShape inputsShape = inputs->getShape();
    TensorShape outputsShape = outputs->getShape();

    // There are four degrees of freedom we can play with in total:
    // N (batch), H (rows), C (channels), and P (ofmap).
    // Each tiling strategy may reduce this down to just three.
    // 1. Start with inputs. Enumerate all shapes that fit.
    // 2. Move on to outputs. Enumerate all shapes that are compatible with
    //    the input shape and fit.
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
        minShape[3] = kVectorSize;
        enum4DTensorTilingConfigs(inputsShape,
                                  maxTileSize,
                                  minShape,
                                  { 1, 1, 1, kVectorSize },
                                  inputConfigs);
    } else if (inputTilingDims == DimNH) {
        std::vector<int> minShape = inputsShape.dims();
        minShape[0] = 1;
        minShape[1] = poolSize.first;
        enum4DTensorTilingConfigs(inputsShape,
                                  maxTileSize,
                                  minShape,
                                  { 1, poolStride.first, 1, 1 },
                                  inputConfigs);
    } else if (inputTilingDims == DimNW) {
        std::vector<int> minShape = inputsShape.dims();
        minShape[0] = 1;
        minShape[2] = poolSize.second;
        enum4DTensorTilingConfigs(inputsShape,
                                  maxTileSize,
                                  minShape,
                                  { 1, 1, poolStride.second, 1 },
                                  inputConfigs);
    } else if (inputTilingDims == DimNHW) {
        std::vector<int> minShape = { 1, poolSize.first, poolSize.second,
                                      inputsShape[3] };
        std::vector<int> strides = { 1, poolStride.first, poolStride.second,
                                     1 };
        enum4DTensorTilingConfigs(
                inputsShape, maxTileSize, minShape, strides, inputConfigs);
    } else if (inputTilingDims == DimNCH) {
        std::vector<int> minShape = { 1, poolSize.first, inputsShape[2],
                                      kVectorSize };
        std::vector<int> strides = { 1, poolStride.first, 1, kVectorSize };
        enum4DTensorTilingConfigs(
                inputsShape, maxTileSize, minShape, strides, inputConfigs);
    } else if (inputTilingDims == DimNCW) {
        std::vector<int> minShape = { 1, inputsShape[1], poolSize.second,
                                      kVectorSize };
        std::vector<int> strides = { 1, 1, poolStride.second, kVectorSize };
        enum4DTensorTilingConfigs(
                inputsShape, maxTileSize, minShape, strides, inputConfigs);
    } else {
        inputConfigs.push_back(inputsShape);
    }
    assert(!inputConfigs.empty() && "No tiling configurations found!");

    // Fill in outputs.
    std::vector<TilingConfig> fullConfigs;
    for (auto it = inputConfigs.begin(); it != inputConfigs.end();
         ++it) {
        TilingConfig config(*it);
        config.outputs = outputsShape;
        config.outputs[0] = config.inputs[0];
        if (needsHwiseTiling(outputTilingDims)) {
            config.outputs[1] = op->calcOutputRows(config.inputs[1]);
        }
        if (needsWwiseTiling(outputTilingDims)) {
            config.outputs[2] = op->calcOutputCols(config.inputs[2]);
        }
        // If inputs and outputs both need channelwise tiling, make the tiles
        // have the same number of channels.
        if (needsCwiseTiling(inputTilingDims) &&
            needsCwiseTiling(outputTilingDims)) {
            config.outputs[3] = config.inputs[3];
        }
        if (config.outputs.storageSize() <= maxTileSize) {
            fullConfigs.push_back(config);
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
    maxIt->outputTilingDims = outputTilingDims;
    return *maxIt;
}

std::array<TiledTensor, 2> TilingOptimizer::doTiling(SmvPoolingOp* op) {
    auto input = op->getInput(SmvPoolingOp::Inputs);
    auto output = op->getOutput(SmvPoolingOp::Outputs);
    TilingConfig tileConfig = TilingOptimizer::computeBasicTileShapes(op);
    int poolRowSize, poolColSize, poolRowStride, poolColStride;
    std::tie(poolRowSize, poolColSize) = op->getPoolingSize();
    std::tie(poolRowStride, poolColStride) = op->getPoolingStride();
    TiledTensor tiledInputs = generateTiledTensor(input,
                                                  tileConfig.inputs,
                                                  op,
                                                  poolRowSize,
                                                  poolColSize,
                                                  poolRowStride,
                                                  poolColStride);
    TiledTensor tiledOutputs =
            generateTiledTensor(output, tileConfig.outputs, op);
    return { tiledInputs, tiledOutputs };
}

}  // namespace pool
}  // namespace smv
}  // namespace smaug
