#include <algorithm>

#include "core/backend.h"
#include "operators/common.h"
#include "operators/smv/smv_pooling_op.h"
#include "operators/smv/smv_pooling_tiling.h"
#include "utility/debug_stream.h"

namespace smaug {
namespace smv {
namespace pool {

// Find the best set of dimensions to tile a given tensor shape.
//
// The goal is to divide up a tensor into tiles that each are <= maxTileSize
// elements.  Assuming that the input tensor data layout is NHWC, the
// preferences for tiling dimensions are as follows:
//   1) No tiling.
//   2) Dim-N tiling. N is the input batch.
//   3) Dim-NC tiling. After tiling by N, tile channelwise. Do not tile in HW.
//   4) Dim-NH tiling. After tiling by N, tile rowwise. Do not tile in WC.
//   5) Dim-NCH tiling. After tiling by N and channel dimensions, tile rowwise.
//      Do not tile in W.
//
// For options 2-5, a minimum size for each dimension can be specified via
// minN, minH, and minC.
TilingDims TilingOptimizer::findBestTilingDims(const TensorShape& shape,
                                               int maxTileSize,
                                               int minN,
                                               int minH,
                                               int minC) {
    if (shape.storageSize() <= maxTileSize)
        return TilingDims::None;
    minN = std::min(shape[0], minN);
    minH = std::min(shape[1], minH);
    // C is the last dimension, so we need to apply padding.
    minC = std::min(shape[3] + shape.getPadding(3), minC);
    int sizePerN = shape.storageSize() / shape[0];
    if (sizePerN * minN <= maxTileSize)
        return TilingDims::DimN;
    if (sizePerN * (minC * 1.0 / shape[3]) <= maxTileSize)
        return TilingDims::DimNC;
    if (sizePerN * (minH * 1.0 / shape[1]) <= maxTileSize)
        return TilingDims::DimNH;
    if (sizePerN * (minC * 1.0 / shape[3]) * (minH * 1.0 / shape[1]) <=
        maxTileSize)
        return TilingDims::DimNCH;
    std::cerr << "[ERROR]: Unable to find a supported set of tiling dimensions "
                 "for tensor with shape " << shape << "!\n";
    assert(false && "Unable to find valid tiling dimensions.");
    return TilingDims::Invalid;
}

// Determine the best tiling dimensions for running pooling on SMV.
//
// This function imposes some additional constraints on the tiling dimensions,
// in that certain combinations of input/output tiling dimensions are
// not allowed in the interest of tiling code complexity.
//
// Returns:
//   A 2-element array of TilingDims enums (inputs, outputs).
std::array<TilingDims, 2> TilingOptimizer::determineBestTilingDims(
        Tensor* inputs,
        Tensor* outputs,
        int maxTileSize,
        std::pair<int, int> poolSize) {
    // Determine the best tiling strategy for each of inputs and outputs. Don't
    // try to figure out the actual tile sizes yet.
    TilingDims bestInputTilingDims = findBestTilingDims(inputs->getShape(),
                                                        maxTileSize,
                                                        1,
                                                        poolSize.first,
                                                        kVectorSize);
    TilingDims bestOutputTilingDims = findBestTilingDims(
            outputs->getShape(), maxTileSize, 1, 1, kVectorSize);

    // Apply some constraints to simplify tiling logic.
    //
    // If inputs require rowwise tiling, then outputs also require rowwise
    // tiling. Strictly speaking this is not necessarily required but it will
    // greatly simplify memory management.
    if (needsHwiseTiling(bestInputTilingDims)) {
        if (needsCwiseTiling(bestOutputTilingDims))
            bestOutputTilingDims = DimNCH;
        else
            bestOutputTilingDims = DimNH;
    }

    return { bestInputTilingDims, bestOutputTilingDims };
}

// Determine the best basic tiling shape for this pooling layer.
//
// The algorithm first determines the dimensions along which the inputs and
// outputs will be tiled. Then based on those dimensions, we enumerate all
// possible basic tile shapes for inputs and outputs. A **basic** shape is the
// shape that all but potentially the last tile along a set of dimensions will
// use. This duo of tile shapes defines a TilingConfig. The TilingConfig
// that maximizes the total combined size of input and output tiles is
// chosen as the best.
//
// To limit the number of possibilities, we only enumerate each dimension in
// certain increments. For example, input channels are only enumerated in
// multiples of kVectorSize.
//
// This algorithm assumes that the maximum tile size for inputs and outputs are
// all the same and that they will reside in separate scratchpads (no sharing).
//
// Args:
//    op: The SMV pooling operator. All tensors must have been created with
//        createAllTensors() prior to calling this function.
// Returns:
//    The TilingConfig that describes the best tiling shapes.
//
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

    dout(2) << "Tiling dimensions chosen: \n"
            << "  input: " << inputTilingDims
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
    // TODO: the tiling logic here shares a lot in common with the convolution
    // operator. We should be able to refactor the code.
    std::list<TilingConfig> inputConfigs;
    if (inputTilingDims == DimN) {
        for (int n = 1; n <= inputsShape[0]; n++) {
            TilingConfig config;
            config.inputs = inputsShape;
            config.inputs[0] = n;
            if (config.inputs.storageSize() <= maxTileSize)
                inputConfigs.push_back(config);
            else
                break;
        }
    } else if (inputTilingDims == DimNC) {
        for (int n = 1; n <= inputsShape[0]; n++) {
            int minChannels = std::min(kVectorSize, inputsShape[3]);
            for (int c = minChannels; c <= inputsShape[3]; c += kVectorSize) {
                TilingConfig config;
                config.inputs = inputsShape;
                config.inputs[0] = n;
                config.inputs[3] = c;
                if (config.inputs.storageSize() <= maxTileSize)
                    inputConfigs.push_back(config);
                else
                    break;
            }
        }
    } else if (inputTilingDims == DimNH) {
        for (int n = 1; n <= inputsShape[0]; n++) {
            int minRows = std::min(inputsShape[1], poolSize.first);
            for (int r = minRows; r <= inputsShape[1]; r += poolStride.first) {
                TilingConfig config;
                config.inputs = inputsShape;
                config.inputs[0] = n;
                config.inputs[1] = r;
                if (config.inputs.storageSize() <= maxTileSize)
                    inputConfigs.push_back(config);
                else
                    break;
            }
        }
    } else if (inputTilingDims == DimNCH) {
        int minChannels = std::min(kVectorSize, inputsShape[3]);
        int minRows = std::min(inputsShape[1], poolSize.first);
        for (int n = 1; n <= inputsShape[0]; n++) {
            for (int c = minChannels; c <= inputsShape[3]; c += kVectorSize) {
                for (int r = minRows; r <= inputsShape[1];
                     r += poolStride.first) {
                    TilingConfig config;
                    config.inputs = TensorShape({ n, r, inputsShape[2], c },
                                                inputsShape.getLayout(),
                                                SmvBackend::Alignment);
                    if (config.inputs.storageSize() <= maxTileSize)
                        inputConfigs.push_back(config);
                    else
                        break;
                }
            }
        }
    } else {
        TilingConfig config;
        config.inputs = inputsShape;
        inputConfigs.push_back(config);
    }
    assert(!inputConfigs.empty() && "No tiling configurations found!");

    // Fill in outputs.
    std::vector<TilingConfig> fullConfigs;
    for (auto it = inputConfigs.begin(); it != inputConfigs.end();
         ++it) {
        TilingConfig config = *it;
        config.outputs = outputsShape;
        config.outputs[0] = config.inputs[0];
        if (needsHwiseTiling(outputTilingDims)) {
            config.outputs[1] = op->calcOutputRows(config.inputs[1]);
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
    dout(2) << "Number of possible tiling configs: " << fullConfigs.size()
            << "\n";
    for (auto& config: fullConfigs) {
        dout(2) << "  inputs: " << config.inputs
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
    std::vector<int> inputHalos{ 0, std::max(poolRowSize - poolRowStride, 0),
                                 std::max(poolColSize - poolColStride, 0), 0 };
    TiledTensor tiledInputs = generateTiledTensor(
            input, tileConfig.inputs, inputHalos, op->getWorkspace());
    TiledTensor tiledOutputs = generateTiledTensor(
            output, tileConfig.outputs, { 0, 0, 0, 0 }, op->getWorkspace());
    return { tiledInputs, tiledOutputs };
}

}  // namespace pool
}  // namespace smv
}  // namespace smaug
