#include "core/backend.h"
#include "operators/common.h"
#include "operators/smv/smv_batch_norm_op.h"
#include "operators/smv/smv_batch_norm_tiling.h"
#include "operators/smv/smv_kernels.h"
#include "utility/debug_stream.h"

namespace smaug {
namespace smv {
namespace bn {

const int kVectorSize = 8;

}  // namespace bn
}  // namespace smv

// The tile dispatcher for post-FC batch norms. The tile iteration is in the
// following order:
// 1) N: batch-wise tiles in the inputs.
// 2) A: activation-wise tiles in the inputs/weights.
void SmvBatchNormOp::runNA(TiledTensor& inputs,
                           TiledTensor& weights,
                           TiledTensor& outputs) {
    int inputNumTiles = inputs.getShape()[0];
    int inputActTiles = inputs.getShape()[1];
    int weightActTiles = weights.getShape()[1];
    auto inputIdx = inputs.startIndex();
    auto weightIdx = weights.startIndex();
    auto outputIdx = outputs.startIndex();
    for (int N = 0; N < inputNumTiles; N++) {
        int iC = 0, wC = 0;
        // This keeps track of the activation offset of the inputs.
        int actOffset = 0;
        while (iC < inputActTiles && wC < weightActTiles) {
            dout(1) << "Input: " << inputIdx(N, iC)
                    << ", weight: " << weightIdx(0, wC)
                    << ", output: " << outputIdx(N, iC) << "\n";
            Tensor* inputTile = inputs[inputIdx(N, iC)];
            Tensor* weightsTile = weights[weightIdx(0, wC)];
            Tensor* outputTile = outputs[outputIdx(N, iC)];
            const TensorShape& inputShape = inputTile->getShape();
            const TensorShape& weightsShape = weightsTile->getShape();
            const TensorShape& outputShape = outputTile->getShape();
            mapArrayToAccel(smv::kBatchNormHw, "host_inputs",
                            inputTile->data<float16>(),
                            inputShape.storageSize() * sizeof(float16));
            mapArrayToAccel(smv::kBatchNormHw, "host_weights",
                            weightsTile->data<float16>(),
                            weightsShape.storageSize() * sizeof(float16));
            mapArrayToAccel(smv::kBatchNormHw, "host_results",
                            outputTile->data<float16>(),
                            outputShape.storageSize() * sizeof(float16));
            int inputDims[2] = { inputShape[0], inputShape[1] };
            // If the input and weight tiles belong to the same channel
            // group, then their data will be loaded at the same time into
            // the spads, so we start from the beginning of the tile.
            // Otherwise, we start from the last place we left off from.
            int actStart = (iC == wC) ? 0 : actOffset;
            // Send the results back to host memory when we finish the weights.
            bool sendOutputs = iC == wC || wC == weightActTiles - 1;

            invokeKernel(smv::kBatchNormHw, smv_batch_norm_post_fc_nc_vec_fxp,
                         inputTile->data<float16>(),
                         weightsTile->data<float16>(),
                         outputTile->data<float16>(), smv::spad0, smv::spad1,
                         smv::spad2, inputDims, weightsShape[1],
                         inputShape.getPadding(1), actStart, sendOutputs,
                         actInfo.function, actInfo.params);

            actOffset += weightsTile->getShape()[1];
            if (inputActTiles == weightActTiles) {
                iC++;
                wC++;
            } else if (inputActTiles == 1) {
                wC++;
            } else {
                assert(false && "The input/weight tiles can have different "
                                "number of channels only when the inputs "
                                "don't need activation-wise tiling.");
            }
        }
    }
}

// The tile dispatcher for post-convolution batch norms. The tile iteration is
// in the following order:
// 1) N: batch-wise tiles in the inputs.
// 2) W: column-wise tiles in the inputs.
// 3) C: channel-wise tiles in the inputs.
// TODO: Add row-wise tiling if we need it later.
void SmvBatchNormOp::runNWC(TiledTensor& inputs,
                            TiledTensor& weights,
                            TiledTensor& outputs) {
    // Ordinarily, we don't need to tile the weights.
    assert(weights.size() == 1);
    int inputNumTiles = inputs.getShape()[0];
    int inputColTiles = inputs.getShape()[2];
    int inputChanTiles = inputs.getShape()[3];
    auto inputIdx = inputs.startIndex();
    auto outputIdx = outputs.startIndex();
    Tensor* weightTile = weights[0];
    const TensorShape& weightShape = weightTile->getShape();
    mapArrayToAccel(smv::kBatchNormHw, "host_weights",
                    weightTile->data<float16>(),
                    weightShape.storageSize() * sizeof(float16));
    for (int N = 0; N < inputNumTiles; N++) {
        for (int W = 0; W < inputColTiles; W++) {
            // This keeps track of the channel offset of the inputs.
            int ifmapOffset = 0;
            for (int C = 0; C < inputChanTiles; C++) {
                dout(1) << "Input: " << inputIdx(N, 0, W, C) << ", Weight: 0"
                        << ", output: " << outputIdx(N, 0, W, C) << "\n";
                Tensor* inputTile = inputs[inputIdx(N, 0, W, C)];
                Tensor* outputTile = outputs[outputIdx(N, 0, W, C)];
                const TensorShape& inputShape = inputTile->getShape();
                const TensorShape& outputShape = outputTile->getShape();
                mapArrayToAccel(smv::kBatchNormHw, "host_inputs",
                                inputTile->data<float16>(),
                                inputShape.storageSize() * sizeof(float16));
                mapArrayToAccel(smv::kBatchNormHw, "host_results",
                                outputTile->data<float16>(),
                                outputShape.storageSize() * sizeof(float16));
                int inputDims[4] = { inputShape[0], inputShape[1],
                                     inputShape[2], inputShape[3] };

                invokeKernel(
                        smv::kBatchNormHw,
                        smv_batch_norm_post_conv_nhwc_vec_fxp,
                        inputTile->data<float16>(), weightTile->data<float16>(),
                        outputTile->data<float16>(), smv::spad0, smv::spad1,
                        smv::spad2, inputDims, weightShape[1],
                        inputShape.getPadding(3), weightShape.getPadding(1),
                        ifmapOffset, actInfo.function, actInfo.params);
                ifmapOffset += inputShape[3];
            }
        }
    }
}

void SmvBatchNormOp::run() {
    using namespace smaug::smv::bn;
    auto input = getInput(Inputs);
    auto mean = getInput(Mean);
    auto variance = getInput(Variance);
    auto gamma = getInput(Gamma);
    auto beta = getInput(Beta);
    auto output = getOutput(Outputs);
    const TensorShape& inputShape = input->getShape();
    const TensorShape& kernelShape = mean->getShape();
    const TensorShape& outputShape = output->getShape();
    bool isPostConv = (input->ndims() == 4);
    dout(2) << *mean << "\n";
    dout(2) << *variance<< "\n";
    dout(2) << *gamma << "\n";
    dout(2) << *beta << "\n";

    // This function will tile (if necessary) the input/weight/output tensors
    // of the batch norm operator into smaller tensor tiles so that each tile
    // can fit in the corresponding scratchpad of the accelerator. It merges
    // the four weights tensors into one and does tiling on it.
    std::array<TiledTensor, 3> tiledTensors = TilingOptimizer::doTiling(this);
    if (isPostConv) {
        assert(inputShape.getLayout() == DataLayout::NHWC);
        assert(outputShape.getLayout() == DataLayout::NHWC);
        runNWC(tiledTensors[0], tiledTensors[1], tiledTensors[2]);
    } else {
        assert(inputShape.getLayout() == DataLayout::NC);
        assert(outputShape.getLayout() == DataLayout::NC);
        runNA(tiledTensors[0], tiledTensors[1], tiledTensors[2]);
    }
    untileTiledTensor(tiledTensors[2], output);
}

}  // namespace smaug
