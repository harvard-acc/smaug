#include "catch.hpp"
#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/core/smaug_test.h"
#include "smaug/operators/depthwise_convolution_op.h"

using namespace smaug;

TEST_CASE_METHOD(SmaugTest, "Reference convolution operator", "[refop]") {
    auto convOp =
            new DepthwiseConvolutionOp<ReferenceBackend>("conv", workspace());
    TensorShape inputShape({ 1, 2, 5, 5 }, DataLayout::NCHW);
    Tensor* input = new Tensor("input", inputShape);
    input->allocateStorage<float>();
    // Input data looks like:
    input->fillData<float>({ 1, 1, 1, 1, 1,  // chan 1
                             2, 2, 2, 2, 2,
                             3, 3, 3, 3, 3,
                             4, 4, 4, 4, 4,
                             5, 5, 5, 5, 5,
                             2, 2, 2, 2, 2,  // chan 2
                             3, 3, 3, 3, 3,
                             4, 4, 4, 4, 4,
                             5, 5, 5, 5, 5,
                             6, 6, 6, 6, 6,
                             });
    workspace()->addTensor(input);
    convOp->setInput(input, 0);
    SECTION("Same padding, 3x3 kernel, stride 1") {
        convOp->setPadding(SamePadding);
        convOp->setWeightDims(3, 3, 1);
        convOp->setStride(1, 1);
        convOp->createAllTensors();
        allocateAllTensors<float>(convOp);
        // Weight kernel:
        // 1  2  3  // chan 1
        // 4  5  6
        // 7  8  9
        // 2  3  4  // chan 2
        // 5  6  7
        // 8  9  10
        auto weightsTensor = convOp->getInput(1);
        weightsTensor->fillData<float>(
                { 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 10 });

        convOp->run();

        // Expected output (with zero padding):
        //
        // Channel 1: see ordinary conv op test.
        // Channel 2:
        // Row 1:
        // 0(2+3+4) + 0(5) + 2(6+7) + 0(8) + 3(9+10) = 83
        // 0(2+3+4) + 2(5+6+7) + 3(8+9+10) = 117
        // 0(2+3+4) + 2(5+6) + 0(7) + 3(8+9) + 0(10) = 73
        //
        // And so on...
        std::vector<float> expectedValues{ 45,  63,  63,  63,  39,
                                           78,  108, 108, 108, 66,
                                           111, 153, 153, 153, 93,
                                           144, 198, 198, 198, 120,
                                           75,  99,  99,  99,  57,
                                           83, 117, 117, 117, 73,
                                           129, 180, 180, 180, 111,
                                           168, 234, 234, 234, 144,
                                           207, 288, 288, 288, 177,
                                           113, 153, 153, 153, 91 };
        auto outputsTensor = convOp->getOutput(0);
        verifyOutputs(outputsTensor, expectedValues);
    }

    SECTION("Valid padding, 3x3 kernel, stride 1") {
        convOp->setPadding(ValidPadding);
        convOp->setWeightDims(3, 3, 1);
        convOp->setStride(1, 1);
        convOp->createAllTensors();
        allocateAllTensors<float>(convOp);
        // Weight kernel:
        // 1  2  3
        // 4  5  6
        // 7  8  9
        auto weightsTensor = convOp->getInput(1);
        weightsTensor->fillData<float>(
                { 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 10 });

        convOp->run();

        // Expected output:
        //
        // 1(1+2+3) + 2(4+5+6) + 3(7+8+9) = 108
        // 2(1+2+3) + 3(4+5+6) + 4(7+8+9) = 153
        // 3(1+2+3) + 4(4+5+6) + 5(7+8+9) = 198
        std::vector<float> expectedValues{ 108, 108, 108,
                                           153, 153, 153,
                                           198, 198, 198,
                                           180, 180, 180,
                                           234, 234, 234,
                                           288, 288, 288 };
        auto outputsTensor = convOp->getOutput(0);
        verifyOutputs(outputsTensor, expectedValues);
    }
}

