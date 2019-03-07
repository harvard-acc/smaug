#include "catch.hpp"
#include "core/backend.h"
#include "core/tensor.h"
#include "core/smaug_test.h"
#include "operators/convolution_op.h"

using namespace smaug;

TEST_CASE_METHOD(SmaugTest, "Reference convolution operator", "[refop]") {
    auto convOp = new ConvolutionOp<ReferenceBackend>("conv", workspace());
    TensorShape inputShape({ 1, 1, 5, 5 }, DataLayout::NCHW);
    Tensor* input = new Tensor("input", inputShape);
    input->allocateStorage<float>();
    // Input data looks like:
    input->fillData<float>({ 1, 1, 1, 1, 1,
                             2, 2, 2, 2, 2,
                             3, 3, 3, 3, 3,
                             4, 4, 4, 4, 4,
                             5, 5, 5, 5, 5 });
    workspace()->addTensor(input);
    convOp->setInput(input, 0);
    SECTION("Same padding, 3x3 kernel, stride 1") {
        convOp->setPadding(SamePadding);
        convOp->setWeightDims(3, 3, 1);
        convOp->setStride(1, 1);
        convOp->createAllTensors();
        allocateAllTensors<float>(convOp);
        // Weight kernel:
        // 1  2  3
        // 4  5  6
        // 7  8  9
        auto weightsTensor = convOp->getInput(1);
        weightsTensor->fillData<float>({1, 2, 3, 4, 5, 6, 7, 8, 9});

        convOp->run();

        // Expected output (with zero padding):
        //
        // Row 1:
        // 0(1+2+3) + 0(4) + 1(5+6) + 0(7) + 2(8+9) = 45
        // 0(1+2+3) + 1(4+5+6) + 2(7+8+9) = 63
        // 0(1+2+3) + 1(4+5) + 0(6) + 2(7+8) + 0(9) = 39
        //
        // Row 2:
        // 0(1) + 1(2+3) + 0(4) + 2(5+6) + 0(7) + 3(8+9) = 78
        // 1(1+2+3) + 2(4+5+6) + 3(7+8+9) = 108
        // 1(1+2) + 0(3) + 2(4+5) + 0(6) + 3(7+8) + 0(9) = 66
        //
        // Row 3:
        // 0(1) + 2(2+3) + 0(4) + 3(5+6) + 0(7) + 4(8+9) = 111
        // 2(1+2+3) + 3(4+5+6) + 4(7+8+9) = 153
        // 2(1+2) + 0(3) + 3(4+5) + 0(6) + 4(7+8) + 0(9) = 93
        //
        // Row 4:
        // 0(1) + 3(2+3) + 0(4) + 4(5+6) + 0(7) + 5(8+9) = 144
        // 3(1+2+3) + 4(4+5+6) + 5(7+8+9) = 198
        // 3(1+2) + 0(3) + 4(4+5) + 0(6) + 5(7+8) + 0(9) = 114
        //
        // Row 5:
        // 0(1) + 4(2+3) + 0(4) + 5(5+6) + 0(7+8+9) = 75
        // 4(1+2+3) + 5(4+5+6) + 0(7+8+9) = 99
        // 4(1+2) + 0(3) + 5(4+5) + 0(6) + 0(7+8+9) = 57
        std::vector<float> expectedValues{ 45,  63,  63,  63,  39,
                                           78,  108, 108, 108, 66,
                                           111, 153, 153, 153, 93,
                                           144, 198, 198, 198, 120,
                                           75,  99,  99,  99,  57 };
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
        weightsTensor->fillData<float>({1, 2, 3, 4, 5, 6, 7, 8, 9});

        convOp->run();

        // Expected output:
        //
        // 1(1+2+3) + 2(4+5+6) + 3(7+8+9) = 108
        // 2(1+2+3) + 3(4+5+6) + 4(7+8+9) = 153
        // 3(1+2+3) + 4(4+5+6) + 5(7+8+9) = 198
        std::vector<float> expectedValues{ 108, 108, 108,
                                           153, 153, 153,
                                           198, 198, 198 };
        auto outputsTensor = convOp->getOutput(0);
        verifyOutputs(outputsTensor, expectedValues);
    }
}
