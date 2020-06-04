#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/smv/smv_convolution_op.h"
#include "smaug/operators/smv/smv_convolution_tiling.h"
#include "smaug/operators/smv/smv_kernels.h"
#include "smaug/operators/smv/smv_accel_pool.h"
#include "smaug/utility/debug_stream.h"

namespace smaug {
namespace smv {
namespace conv {

const int kNumPEs = 8;
const int kNumMaccsPerPE = 32;

}  // namespace conv
}  // namespace smv

void SmvConvolutionOp::runNHWC(TiledTensor& inputs,
                               TiledTensor& weights,
                               TiledTensor& outputs) {
    int inputIfmapTiles = inputs.getShape()[0];
    int inputRowTiles = inputs.getShape()[1];
    int inputChanTiles = inputs.getShape()[3];
    int weightOfmapTiles = weights.getShape()[0];
    int weightChanTiles = weights.getShape()[3];
    int outputRowTiles = outputs.getShape()[1];
    int outputChanTiles = outputs.getShape()[3];
    auto inputIdx = inputs.startIndex();
    auto weightIdx = weights.startIndex();
    auto outputIdx = outputs.startIndex();
    std::vector<int> inputPadding = getInputPadding();
    int topPad = inputPadding[0];
    int bottomPad = inputPadding[1];
    int leftPad = inputPadding[2];
    int rightPad = inputPadding[3];
    unsigned accelId = useSystolicArrayWhenAvailable ? smv::kSystolicArrayHw
                                                     : smv::kConvolutionHw;
    SmvAcceleratorPool accelPool(numAcceleratorsAvailable);
    std::vector<int> lastReadInputTileIdx(numAcceleratorsAvailable, -1);
    std::vector<int> lastReadWeightTileIdx(numAcceleratorsAvailable, -1);
    for (int i = 0; i < numAcceleratorsAvailable; i++) {
        setArrayMemTypeIfSimulating(
                accelId + i, "host_inputs", getInputsMemType());
        setArrayMemTypeIfSimulating(
                accelId + i, "host_weights", getWeightsMemType());
        setArrayMemTypeIfSimulating(
                accelId + i, "host_results", getOutputsMemType());
    }
    int currAccelIdx = 0;
    for (int N = 0; N < inputIfmapTiles; N++) {
        for (int H = 0; H < outputRowTiles; H++) {
            int currentTileTopPad = topPad;
            int currentTileBottomPad = bottomPad;
            if (inputRowTiles > 1) {
                if (H == 0) {
                    currentTileBottomPad = 0;
                } else if (H == inputRowTiles - 1) {
                    currentTileTopPad = 0;
                } else {
                    currentTileTopPad = 0;
                    currentTileBottomPad = 0;
                }
            }
            // This is used to specify the padding sizes on the boundaries of
            // the 2D feature maps in an input tile.
            int inputHaloPad[4] = { currentTileTopPad, currentTileBottomPad,
                                    leftPad, rightPad };
            // On one condition, the tiling optimizer allows the weight tile to
            // contain more kernels than the output tile: the weights do not
            // need N-wise tiling (weightOfmapTiles = 1), whereas the output
            // needs channelwise tiling (weightOfmapTiles < outputChanTiles).
            // We will then need multiple kernel invocations to finish the
            // weight tile, where each invocation only consumes part of it. The
            // argument 'kern_start' is used for this: it provides the starting
            // kernel from which the weight tile will be effective.
            bool needOutputIteration = weightOfmapTiles < outputChanTiles;
            int kernStart = 0;
            // This is the number of invocations we need to finish the weight
            // tile. In common scenarios, only one invocation is needed. If we
            // need to iterate the output channels, outputChanTiles invocatons
            // are needed to finish the weight tile.
            int numOutputInvocations =
                    needOutputIteration ? outputChanTiles : 1;
            assert(numOutputInvocations > 1
                           ? weightOfmapTiles == 1
                           : weightOfmapTiles == outputChanTiles);
            for (int W = 0; W < weightOfmapTiles; W++) {
                // We have three loop levels up to this point, the first for
                // input batch-wise tiles iteration, the second for input
                // rowwise tiles iteration, the third for weight N-wise tiles
                // iteration. There is no data dependency among the loop nests
                // involve in these levels, and therefore we can run them
                // in parallel.
                //
                // We have another two loop level beyond this point, one for
                // output channelwise tiles iteration and the other for weight
                // channelwise tiles iteration. We run these loop nests in
                // serial (i.e., on one single accelerator). The ones in the
                // latter loop accumulate results to the same output tile and
                // thus exhibiting data dependency, whereas the former could run
                // in parallel technically, but we will need to reload too much
                // weights for that and therefore I choose not to.
                for (int oC = 0; oC < numOutputInvocations; oC++) {
                    int iC = 0, wC = 0;
                    // This keeps track of the channel offset of the input.
                    int ifmapOffset = 0;
                    int outputTileIdx = outputIdx(N, H, 0, W + oC);
                    Tensor* outputTile = outputs[outputTileIdx];
                    const TensorShape& outputShape = outputTile->getShape();
                    mapArrayToAccel(
                            accelId + currAccelIdx, "host_results",
                            outputTile->data<float16>(),
                            outputShape.storageSize() * sizeof(float16));

                    // The tiling optimizer will make sure that the weight tiles
                    // have the same channel dimension as the input tiles (so
                    // that inputChanTiles = weightChanTiles), except one case
                    // where the input is not tiled channelwise (inputChanTiles
                    // = 1) and the weights are independently tiled channelwise.
                    // In that case, we will need multiple kernel invocations to
                    // finish the weight channelwise tiles, with the same input
                    // channel tile, producing results for the same output
                    // channels.
                    while (iC < inputChanTiles && wC < weightChanTiles) {
                        int inputTileIdx = inputIdx(N, H, 0, iC);
                        int weightTileIdx = weightIdx(W, 0, 0, wC);
                        dout(1) << "Input: " << inputTileIdx
                                << ", weights: " << weightTileIdx
                                << ", output: " << outputTileIdx << "\n";
                        Tensor* inputTile =
                                inputs.getTileWithData(inputTileIdx);
                        Tensor* weightsTile =
                                weights.getTileWithData(weightTileIdx);
                        const TensorShape& inputShape = inputTile->getShape();
                        const TensorShape& weightsShape =
                                weightsTile->getShape();
                        mapArrayToAccel(
                                accelId + currAccelIdx, "host_inputs",
                                inputTile->data<float16>(),
                                inputShape.storageSize() * sizeof(float16));
                        mapArrayToAccel(
                                accelId + currAccelIdx, "host_weights",
                                weightsTile->data<float16>(),
                                weightsShape.storageSize() * sizeof(float16));
                        int inputDims[4] = { inputShape[0], inputShape[1],
                                             inputShape[2], inputShape[3] };
                        int weightsDims[4] = { weightsShape[0], weightsShape[1],
                                               weightsShape[2],
                                               weightsShape[3] };
                        int outputDims[4] = { outputShape[0], outputShape[1],
                                              outputShape[2], outputShape[3] };
                        // The 'ifmap_start' argument of the kernel is for
                        // handling when inputChanTiles < weightChanTiles. It
                        // provides the starting channel of the input tile that
                        // will be effective for computation in the invocation.
                        int ifmapStart = (iC == wC) ? 0 : ifmapOffset;
                        // Since multiple weight channelwise tiles produce the
                        // same output channels, 'accumulate' is set to true to
                        // avoid resetting the result for non-first (wC > 0)
                        // weight channelwise tiles.
                        bool accumulate = wC > 0;
                        // If this is a new input/weight tile, then we need to
                        // read it.
                        bool readInputs = false;
                        if (inputTileIdx !=
                            lastReadInputTileIdx[currAccelIdx]) {
                            readInputs = true;
                            lastReadInputTileIdx[currAccelIdx] = inputTileIdx;
                        }
                        bool readWeights = false;
                        if (weightTileIdx !=
                            lastReadWeightTileIdx[currAccelIdx]) {
                            readWeights = true;
                            lastReadWeightTileIdx[currAccelIdx] = weightTileIdx;
                        }
                        // If we reach the last invocation for the weight
                        // channelwise tiles, the results are finished and need
                        // to be sent back to the host.
                        bool sendResults = wC == weightChanTiles - 1;

                        std::unique_ptr<volatile int> finishFlag;
                        if (useSystolicArrayWhenAvailable) {
                            // Invoke the systolic array if specified.
                            finishFlag = invokeSystolicArrayKernel(
                                    accelId + currAccelIdx,
                                    inputTile->data<float16>(),
                                    weightsTile->data<float16>(),
                                    outputTile->data<float16>(), inputDims,
                                    weightsDims, outputDims,
                                    inputShape.getPadding(3),
                                    weightsShape.getPadding(3),
                                    outputShape.getPadding(3), inputHaloPad,
                                    getRowStride(), ifmapStart, kernStart,
                                    accumulate, readInputs, readWeights,
                                    sendResults, &actInfo);
                        } else {
                            // Otherwise invoke the DLA-like kernel.
                            finishFlag = invokeKernelNoBlock(
                                    currAccelIdx, accelId + currAccelIdx,
                                    smv_conv3d_nhwc_vec_fxp,
                                    inputTile->data<float16>(),
                                    weightsTile->data<float16>(),
                                    outputTile->data<float16>(), smv::spad0,
                                    smv::spad1, smv::spad2, inputDims,
                                    weightsDims, outputDims,
                                    inputShape.getPadding(3),
                                    weightsShape.getPadding(3),
                                    outputShape.getPadding(3), inputHaloPad,
                                    getRowStride(), getColStride(), ifmapStart,
                                    kernStart, accumulate, readInputs,
                                    readWeights, sendResults, actInfo.function,
                                    actInfo.params, &sampling);
                        }
                        accelPool.addFinishFlag(
                                currAccelIdx, std::move(finishFlag));

                        ifmapOffset += weightsTile->getShape()[3];
                        if (inputChanTiles == weightChanTiles) {
                            iC++;
                            wC++;
                        } else if (inputChanTiles == 1) {
                            wC++;
                        } else {
                            assert(false &&
                                   "The input/weight tiles can have different "
                                   "number of channels only when the inputs "
                                   "don't need channelwise tiling.");
                        }
                    }
                    if (needOutputIteration)
                        kernStart += outputShape[3];
                }
                currAccelIdx =
                        accelPool.getNextAvailableAccelerator(currAccelIdx);
            }
        }
    }
    // Before we leave, make sure all the accelerators have finished.
    accelPool.joinAll();
}

std::unique_ptr<volatile int> SmvConvolutionOp::invokeSystolicArrayKernel(
        unsigned accelId,
        float16* inputs,
        float16* weights,
        float16* outputs,
        int inputsDims[4],
        int weightsDims[4],
        int outputsDims[4],
        int inputsPad,
        int weightsPad,
        int outputPad,
        int inputHaloPad[4],
        int stride,
        int ifmapStart,
        int kernStart,
        bool accumulate,
        bool readInputs,
        bool readWeights,
        bool sendResults,
        ActivationInfo* actInfo) {
    // Note that if we are in trace mode, we should skip this gem5 accelerator.
#ifndef TRACE_MODE
    assert(runningInSimulation && "The systolic array must be invoked in "
                                  "simuation.");
    systolic_array_params_t params;
    params.input_base_addr = inputs;
    params.weight_base_addr = weights;
    params.output_base_addr = outputs;
    memcpy(params.input_dims, inputsDims, sizeof(int) * 4);
    memcpy(params.weight_dims, weightsDims, sizeof(int) * 4);
    memcpy(params.output_dims, outputsDims, sizeof(int) * 4);
    params.input_dims[3] += inputsPad;
    params.weight_dims[3] += weightsPad;
    params.output_dims[3] += outputPad;
    params.stride = stride;
    memcpy(params.input_halo_pad, inputHaloPad, sizeof(int) * 4);
    params.ifmap_start = ifmapStart;
    params.kern_start = kernStart;
    params.accum_results = accumulate;
    params.read_inputs = readInputs;
    params.read_weights = readWeights;
    params.send_results = sendResults;
    // The systolic array kernel in gem5 uses the same
    // activation type/params structures.
    memcpy(&params.act_type, &(actInfo->function), sizeof(activation_type));
    memcpy(&params.act_params, &(actInfo->params), sizeof(activation_param_t));
    return std::unique_ptr<volatile int>(
            invokeSystolicArrayAndReturn(accelId, params));
#else
    return nullptr;
#endif
}

void SmvConvolutionOp::tile() {
    // This function will tile (if necessary) the input/weight/output tensors
    // of the convolution operator into smaller tensor tiles so that each tile
    // can fit in the corresponding scratchpad of the accelerator.
    // TODO: A lot of networks have back to back convolutional layers, it would
    // be much more efficient not to retile in between them. That can be
    // achieved by directly sending the output tiles to the next convolutional
    // layer instead of merging them into a single output tensor first. It's
    // sort of operator fusing that two back-to-back convolution operators are
    // tiled only once.
    tiledTensors = smaug::smv::conv::TilingOptimizer::doTiling(this);
}

void SmvConvolutionOp::run() {
    auto input = getInput(Inputs);
    auto kernels = getInput(Kernels);
    auto output = getOutput(Outputs);
    const TensorShape& inputShape = input->getShape();
    const TensorShape& kernelShape = kernels->getShape();
    const TensorShape& outputShape = output->getShape();
    assert(inputShape.getLayout() == DataLayout::NHWC);
    assert(kernelShape.getLayout() == DataLayout::NHWC);
    assert(outputShape.getLayout() == DataLayout::NHWC);
    dout(2) << *kernels << "\n";

    {
        auto stats = gem5::ScopedStats(
                stats::kTensorPrepStart, stats::kTensorPrepEnd);
        tiledTensors[0].copyDataToAllTiles();
        tiledTensors[1].copyDataToAllTiles();
    }

    runNHWC(tiledTensors[0], tiledTensors[1], tiledTensors[2]);

    {
        auto stats = gem5::ScopedStats(
                stats::kTensorFinalStart, stats::kTensorFinalEnd);
        tiledTensors[2].untile();
    }
}

}  // namespace smaug
