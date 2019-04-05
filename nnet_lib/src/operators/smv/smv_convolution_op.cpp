#include "core/backend.h"
#include "operators/common.h"
#include "operators/smv/smv_convolution_op.h"
#include "operators/smv/smv_convolution_tiling.h"
#include "operators/smv/kernels.h"
#include "utility/debug_stream.h"

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
    auto inputIdx = inputs.startIndex();
    auto weightIdx = weights.startIndex();
    auto outputIdx = outputs.startIndex();
    // Padding sizes on the four boundaries of the input 2D feature map. Here we
    // compute default padding sizes if no rowwise tiling is needed on the
    // input. We handle rowwise tiling in the following loop nests.
    int totalRowPad = (getPadding() == SamePadding) ? getWeightRows() - 1 : 0;
    int totalColPad = (getPadding() == SamePadding) ? getWeightCols() - 1 : 0;
    int topPad = FRAC_CEIL(totalRowPad, 2);
    int bottomPad = totalRowPad - topPad;
    int leftPad = FRAC_CEIL(totalColPad, 2);
    int rightPad = totalColPad - leftPad;
    for (int N = 0; N < inputIfmapTiles; N++) {
        for (int H = 0; H < inputRowTiles; H++) {
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
            for (int W = 0; W < weightOfmapTiles; W++) {
                int iC = 0, wC = 0;
                // This keeps track of the channel offset of the input.
                int ifmapOffset = 0;
                // The tiling optimizer will make sure that the weight tiles
                // have the same channel dimension as the input tiles
                // (so that inputChanTiles = weightChanTiles), except one case
                // where the input is not tiled channelwise (inputChanTiles = 1)
                // and the weights are independently tiled channelwise. In that
                // case, we will need multiple kernel invocations to finish the
                // weight channelwise tiles, with the same input channel tile,
                // producing results for the same output channels.
                while (iC < inputChanTiles && wC < weightChanTiles) {
                    std::cout << "Input: " << inputIdx(N, H, 0, iC)
                              << ", weights: " << weightIdx(W, 0, 0, wC)
                              << ", output: " << outputIdx(N, H, 0, W) << "\n";
                    Tensor* inputTile = inputs[inputIdx(N, H, 0, iC)];
                    Tensor* weightsTile = weights[weightIdx(W, 0, 0, wC)];
                    Tensor* outputTile = outputs[outputIdx(N, H, 0, W)];
                    const TensorShape& inputShape = inputTile->getShape();
                    const TensorShape& weightsShape = weightsTile->getShape();
                    const TensorShape& outputShape = outputTile->getShape();
                    int inputDims[4] = { inputShape[0], inputShape[1],
                                         inputShape[2], inputShape[3] };
                    int weightsDims[4] = { weightsShape[0], weightsShape[1],
                                           weightsShape[2], weightsShape[3] };
                    int outputDims[4] = { outputShape[0], outputShape[1],
                                          outputShape[2], outputShape[3] };
                    // The 'ifmap_start' argument of the kernel is for handling
                    // when inputChanTiles < weightChanTiles. It provides the
                    // starting channel of the input tile that will be effective
                    // for computation in the invocation.
                    int ifmapStart = (iC == wC) ? 0 : ifmapOffset;
                    // Since multiple weight channelwise tiles produce the same
                    // output channels, 'accumulate' is set to true to avoid
                    // resetting the result for non-first (wC > 0) weight
                    // channelwise tiles.
                    bool accumulate = wC > 0;
                    smv_conv3d_f32_nhwc_vec_fxp(inputTile->data<float>(),
                                                weightsTile->data<float>(),
                                                outputTile->data<float>(),
                                                inputDims,
                                                weightsDims,
                                                outputDims,
                                                inputShape.getPadding(3),
                                                weightsShape.getPadding(3),
                                                outputShape.getPadding(3),
                                                inputHaloPad,
                                                getRowStride(),
                                                getColStride(),
                                                ifmapStart,
                                                accumulate);

                    ifmapOffset += weightsTile->getShape()[3];
                    if (inputChanTiles == weightChanTiles) {
                        iC++;
                        wC++;
                    } else if (inputChanTiles == 1) {
                        wC++;
                    }
                }
            }
        }
    }
}

void SmvConvolutionOp::run() {
    using namespace smaug::smv::conv;
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

    // This function will tile (if necessary) the input/weight/output tensors
    // of the convolution operator into smaller tensor tiles so that each tile
    // can fit in the corresponding scratchpad of the accelerator.
    // TODO: A lot of networks have back to back convolutional layers, it would
    // be much more efficient not to retile in between them. That can be
    // achieved by directly sending the output tiles to the next convolutional
    // layer instead of merging them into a single output tensor first. It's
    // sort of operator fusing that two back-to-back convolution operators are
    // tiled only once.
    std::array<TiledTensor, 3> tiledTensors = TilingOptimizer::doTiling(this);
    runNHWC(tiledTensors[0], tiledTensors[1], tiledTensors[2]);
    untileTiledTensor(tiledTensors[2], output);
}

}  // namespace smaug
