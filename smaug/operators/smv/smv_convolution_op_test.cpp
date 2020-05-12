#include "catch.hpp"
#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/core/smaug_test.h"
#include "smaug/operators/smv/smv_test_common.h"
#include "smaug/operators/smv/smv_convolution_op.h"
#include "smaug/operators/smv/smv_convolution_tiling.h"

using namespace smaug;

namespace smaug {

class SmvConvolutionOpTest : public SmaugTest {
   public:
    using SmaugTest::SmaugTest;

    Tensor* getReferenceOutput(SmvConvolutionOp* convOp) {
        auto input = convOp->getInput(0);
        auto kernels = convOp->getInput(1);
        auto input32 = convertFp16ToFp32Tensor(input, workspace());
        auto kernels32 = convertFp16ToFp32Tensor(kernels, workspace());

        // A reference convolution operator is used to get the 'correct' output.
        auto refConvOp =
                new ConvolutionOp<ReferenceBackend>("ref_conv", workspace());
        refConvOp->setActivation(convOp->getActivation());
        refConvOp->setPadding(convOp->getPadding());
        refConvOp->setWeightDims(convOp->getWeightRows(),
                                 convOp->getWeightCols(),
                                 convOp->getNumOfmaps());
        refConvOp->setStride(convOp->getRowStride(), convOp->getColStride());
        refConvOp->setInput(input32, 0);
        refConvOp->setInput(kernels32, 1);
        refConvOp->createAllTensors();
        refConvOp->getOutput(0)->allocateStorage<float>();
        refConvOp->run();
        return convertFp32ToFp16Tensor(refConvOp->getOutput(0), workspace());
    }

    void doTest(std::vector<int> inputDims,
                std::vector<int> kernelDims,
                PaddingType padding = SamePadding,
                std::vector<int> strides = { 1, 1 }) {
        auto convOp = new SmvConvolutionOp("conv", workspace());
        convOp->setStride(strides[0], strides[1]);
        convOp->setPadding(padding);
        TensorShape inputShape(inputDims, NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("input", inputShape);
        inputs->allocateStorage<float16>();
        workspace()->addTensor(inputs);
        convOp->setInput(inputs, 0);
        convOp->setWeightDims(kernelDims[1], kernelDims[2], kernelDims[0]);
        createAndFillTensorsWithData<float16>(convOp, fillTensorWithRandomData);
        convOp->tile();
        convOp->run();
        auto outputs = convOp->getOutput(0);
        auto refOutputs = getReferenceOutput(convOp);
        verifyOutputs<float16>(outputs, refOutputs);
    }

    void doFusionTest(
            std::vector<int> inputDims,
            std::vector<int> kernelDims,
            ActivationInfo actInfo = ActivationInfo(activation_type::ELU),
            PaddingType padding = SamePadding,
            std::vector<int> strides = { 1, 1 }) {
        auto convOp = new SmvConvolutionOp("conv", workspace());
        convOp->setActivation(actInfo);
        convOp->setStride(strides[0], strides[1]);
        convOp->setPadding(padding);
        TensorShape inputShape(inputDims, NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("input", inputShape);
        inputs->allocateStorage<float16>();
        workspace()->addTensor(inputs);
        convOp->setInput(inputs, 0);
        convOp->setWeightDims(kernelDims[1], kernelDims[2], kernelDims[0]);
        createAndFillTensorsWithData<float16>(convOp, fillTensorWithRandomData);
        convOp->tile();
        convOp->run();
        auto outputs = convOp->getOutput(0);
        auto refOutputs = getReferenceOutput(convOp);
        verifyOutputs<float16>(outputs, refOutputs);
    }
};

}  // namespace smaug

TEST_CASE_METHOD(SmvConvolutionOpTest,
                 "SMV Tiled Convolution with fused activation",
                 "[smvconv]") {
    auto convOp = new SmvConvolutionOp("conv", workspace());
    // Outputs should be the same size as inputs.

    SECTION("No tiling required") {
        SECTION("Same padding") {
            // Same padding is used by default.
            doFusionTest({ 1, 8, 8, 8 }, { 8, 3, 3, 8 });
        }
        SECTION("Valid padding") {
            doFusionTest({ 1, 8, 8, 8 },
                         { 8, 3, 3, 8 },
                         activation_type::ELU,
                         ValidPadding);
        }
    }

    SECTION("DimN tiled convolution") {
        SECTION("Every weight tile contains 8 kernels") {
            doFusionTest({ 1, 8, 8, 192 }, { 128, 3, 3, 192 });
        }
        SECTION("Every weight tile contains more than 8 kernels") {
            SECTION("Every weight tile contains multiples of 8 kernels") {
                // The weight tiles will contain 56, 56 and 16 kernels
                // respectively.
                doFusionTest({ 1, 8, 8, 32 }, { 128, 3, 3, 32 });
            }
            SECTION("Weight tile contains non-multiples of 8 kernels") {
                // The weight tiles will contain 50 kernels.
                doFusionTest({ 1, 8, 8, 32 }, { 50, 3, 3, 32 });
            }
        }
    }

    SECTION("DimNH tiled convolution") {
        SECTION("Inputs DimNH tiled, No need to tile the weights") {
            SECTION("Same padding") {
                doFusionTest({ 1, 32, 32, 32 }, { 8, 3, 3, 32 });
            }
            SECTION("Valid padding") {
                doFusionTest({ 1, 32, 32, 32 },
                             { 8, 3, 3, 32 },
                             activation_type::ELU,
                             ValidPadding);
            }
        }
        SECTION("Inputs DimNH tiled, weights DimN tiled") {
            SECTION("5x5 kernel size") {
                doFusionTest({ 1, 32, 32, 32 }, { 128, 5, 5, 32 });
            }
            SECTION("2x2 kernel size") {
                doFusionTest({ 1, 32, 32, 32 }, { 256, 2, 2, 32 });
            }
        }
        // The difference between this and the previous one is the tiling in the
        // weights due to the input channels.
        SECTION("Inputs DimNH tiled, weights DimNC tiled") {
            doFusionTest({ 1, 64, 16, 256 }, { 128, 4, 4, 256 });
        }
    }

    SECTION("DimNC tiled convolution") {
        SECTION("Input tile and weight tile have the same channel dimension.") {
            SECTION("Both have 1 channelwise tile.") {
                doFusionTest({ 1, 16, 8, 64 }, { 8, 5, 5, 64 });
            }
            SECTION("Both have 4 channelwise tiles.") {
                doFusionTest({ 1, 16, 16, 256 }, { 8, 5, 5, 256 });
            }
        }
        SECTION("Inputs are not tiled channelwise, weights have 2 channelwise "
                "tiles") {
            doFusionTest({ 1, 8, 8, 256 }, { 8, 3, 3, 256 });
        }
        SECTION("Inputs are not tiled channelwise, weights have 3 channelwise "
                "tiles") {
            doFusionTest({ 1, 4, 4, 512 }, { 8, 3, 3, 512 });
        }
        SECTION("Inputs and weights don't need tiling, outputs need DimNC "
                "tiling") {
            TensorShape inputShape(
                    { 1, 32, 32, 8 }, DataLayout::NHWC, SmvBackend::Alignment);
            Tensor* inputs = new Tensor("inputs", inputShape);
            workspace()->addTensor(inputs);
            convOp->setInput(inputs, 0);
            SECTION("16 output tiles") {
                doFusionTest({ 1, 32, 32, 8 }, { 256, 1, 1, 8 });
            }
            SECTION("8 output tiles") {
                doFusionTest({ 1, 32, 32, 8 }, { 128, 2, 2, 8 });
            }
            SECTION("4 output tiles") {
                doFusionTest({ 1, 32, 32, 8 }, { 64, 3, 3, 8 });
            }
        }
    }

    SECTION("DimNCH tiled convolution") {
        SECTION("Inputs DimNCH tiling: 3 tiles rowwise, 6 tiles channelwise") {
            doFusionTest({ 1, 32, 32, 192 }, { 32, 4, 4, 192 });
        }
        SECTION("Inputs DimNCH tiling: 9 tiles rowwise, 6 tiles channelwise") {
            doFusionTest({ 1, 64, 64, 192 }, { 32, 2, 2, 192 });
        }
        SECTION("Inputs DimNCH tiling: 43 tiles rowwise, 6 tiles channelwise") {
            doFusionTest({ 1, 128, 128, 192 }, { 32, 2, 2, 192 });
        }
    }
}

TEST_CASE_METHOD(SmvConvolutionOpTest,
                 "SMV Tiled Convolution without fused activation",
                 "[smvconv]") {
    auto convOp = new SmvConvolutionOp("conv", workspace());
    // Outputs should be the same size as inputs.

    SECTION("No tiling required") {
        SECTION("Same padding") {
            // Same padding is used by default.
            doTest({ 1, 8, 8, 8 }, { 8, 3, 3, 8 });
        }
        SECTION("Valid padding") {
            doTest({ 1, 8, 8, 8 }, { 8, 3, 3, 8 }, ValidPadding);
        }
    }

    SECTION("DimN tiled convolution") {
        SECTION("Every weight tile contains 8 kernels") {
            doTest({ 1, 8, 8, 192 }, { 128, 3, 3, 192 });
        }
        SECTION("Every weight tile contains more than 8 kernels") {
            SECTION("Every weight tile contains multiples of 8 kernels") {
                // The weight tiles will contain 56, 56 and 16 kernels
                // respectively.
                doTest({ 1, 8, 8, 32 }, { 128, 3, 3, 32 });
            }
            SECTION("Weight tile contains non-multiples of 8 kernels") {
                // The weight tiles will contain 50 kernels.
                doTest({ 1, 8, 8, 32 }, { 50, 3, 3, 32 });
            }
        }
    }

    SECTION("DimNH tiled convolution") {
        SECTION("Inputs DimNH tiled, No need to tile the weights") {
            SECTION("Same padding") {
                doTest({ 1, 32, 32, 32 }, { 8, 3, 3, 32 });
            }
            SECTION("Valid padding") {
                doTest({ 1, 32, 32, 32 }, { 8, 3, 3, 32 }, ValidPadding);
            }
        }
        SECTION("Inputs DimNH tiled, weights DimN tiled") {
            SECTION("5x5 kernel size") {
                doTest({ 1, 32, 32, 32 }, { 128, 5, 5, 32 });
            }
            SECTION("2x2 kernel size") {
                doTest({ 1, 32, 32, 32 }, { 256, 2, 2, 32 });
            }
        }
        // The difference between this and the previous one is the tiling in the
        // weights due to the input channels.
        SECTION("Inputs DimNH tiled, weights DimNC tiled") {
            doTest({ 1, 64, 16, 256 }, { 128, 4, 4, 256 });
        }
    }

    SECTION("DimNC tiled convolution") {
        SECTION("Input tile and weight tile have the same channel dimension.") {
            SECTION("Both have 1 channelwise tile.") {
                doTest({ 1, 16, 8, 64 }, { 8, 5, 5, 64 });
            }
            SECTION("Both have 4 channelwise tiles.") {
                doTest({ 1, 16, 16, 256 }, { 8, 5, 5, 256 });
            }
        }
        SECTION("Inputs are not tiled channelwise, weights have 2 channelwise "
                "tiles") {
            doTest({ 1, 8, 8, 256 }, { 8, 3, 3, 256 });
        }
        SECTION("Inputs are not tiled channelwise, weights have 3 channelwise "
                "tiles") {
            doTest({ 1, 4, 4, 512 }, { 8, 3, 3, 512 });
        }
        SECTION("Inputs and weights don't need tiling, outputs need DimNC "
                "tiling") {
            TensorShape inputShape(
                    { 1, 32, 32, 8 }, DataLayout::NHWC, SmvBackend::Alignment);
            Tensor* inputs = new Tensor("inputs", inputShape);
            workspace()->addTensor(inputs);
            convOp->setInput(inputs, 0);
            SECTION("16 output tiles") {
                doTest({ 1, 32, 32, 8 }, { 256, 1, 1, 8 });
            }
            SECTION("8 output tiles") {
                doTest({ 1, 32, 32, 8 }, { 128, 2, 2, 8 });
            }
            SECTION("4 output tiles") {
                doTest({ 1, 32, 32, 8 }, { 64, 3, 3, 8 });
            }
        }
    }

    SECTION("DimNCH tiled convolution") {
        SECTION("Inputs DimNCH tiling: 3 tiles rowwise, 6 tiles channelwise") {
            doTest({ 1, 32, 32, 192 }, { 32, 4, 4, 192 });
        }
        SECTION("Inputs DimNCH tiling: 9 tiles rowwise, 6 tiles channelwise") {
            doTest({ 1, 64, 64, 192 }, { 32, 2, 2, 192 });
        }
        SECTION("Inputs DimNCH tiling: 43 tiles rowwise, 6 tiles channelwise") {
            doTest({ 1, 128, 128, 192 }, { 32, 2, 2, 192 });
        }
    }
}

TEST_CASE_METHOD(SmvConvolutionOpTest, "Stride size tests", "[smvconv]") {
    SECTION("2x2 strides") {
        SECTION("1x1 kernels") {
            doTest({ 1, 56, 56, 256 },
                   { 512, 1, 1, 256 },
                   SamePadding,
                   { 2, 2 });
        }
        SECTION("3x3 kernels") {
            doTest({ 1, 64, 64, 32 }, { 16, 3, 3, 32 }, ValidPadding, { 2, 2 });
        }
        SECTION("5x5 kernels") {
            doTest({ 1, 225, 225, 8 }, { 64, 5, 5, 8 }, SamePadding, { 2, 2 });
        }
        SECTION("7x7 kernels") {
            doTest({ 1, 225, 225, 8 }, { 64, 7, 7, 8 }, SamePadding, { 2, 2 });
        }
    }

    SECTION("3x3 strides") {
        SECTION("1x1 kernels") {
            doTest({ 1, 56, 56, 256 },
                   { 512, 1, 1, 256 },
                   SamePadding,
                   { 3, 3 });
        }
        SECTION("3x3 kernels") {
            doTest({ 1, 64, 64, 32 }, { 16, 3, 3, 32 }, ValidPadding, { 3, 3 });
        }
        SECTION("5x5 kernels") {
            doTest({ 1, 225, 225, 8 }, { 64, 5, 5, 8 }, SamePadding, { 3, 3 });
        }
        SECTION("7x7 kernels") {
            doTest({ 1, 225, 225, 8 }, { 64, 7, 7, 8 }, SamePadding, { 3, 3 });
        }
    }
}
