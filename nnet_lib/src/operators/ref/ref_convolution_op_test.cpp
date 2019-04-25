#include "catch.hpp"
#include "core/backend.h"
#include "core/tensor.h"
#include "core/smaug_test.h"
#include "operators/convolution_op.h"

using namespace smaug;

TEST_CASE_METHOD(SmaugTest, "Reference convolution operator", "[refop]") {
    auto convOp = new ConvolutionOp<ReferenceBackend>("conv", workspace());

    SECTION("NCHW layout") {
        TensorShape inputShape({ 1, 1, 5, 5 }, DataLayout::NCHW);
        Tensor* input = new Tensor("input", inputShape);
        input->allocateStorage<float>();
        // Input data looks like:
        input->fillData<float>({
                1, 1, 1, 1, 1,  // Row 0
                2, 2, 2, 2, 2,  // Row 1
                3, 3, 3, 3, 3,  // Row 2
                4, 4, 4, 4, 4,  // Row 3
                5, 5, 5, 5, 5   // Row 4
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
            // 1  2  3
            // 4  5  6
            // 7  8  9
            auto weightsTensor = convOp->getInput(1);
            weightsTensor->fillData<float>({ 1, 2, 3, 4, 5, 6, 7, 8, 9 });

            convOp->run();

            // Expected output (with zero padding):
            //
            // Row 0:
            // 0(1+2+3) + 0(4) + 1(5+6) + 0(7) + 2(8+9) = 45
            // 0(1+2+3) + 1(4+5+6) + 2(7+8+9) = 63
            // 0(1+2+3) + 1(4+5) + 0(6) + 2(7+8) + 0(9) = 39
            //
            // Row 1:
            // 0(1) + 1(2+3) + 0(4) + 2(5+6) + 0(7) + 3(8+9) = 78
            // 1(1+2+3) + 2(4+5+6) + 3(7+8+9) = 108
            // 1(1+2) + 0(3) + 2(4+5) + 0(6) + 3(7+8) + 0(9) = 66
            //
            // Row 2:
            // 0(1) + 2(2+3) + 0(4) + 3(5+6) + 0(7) + 4(8+9) = 111
            // 2(1+2+3) + 3(4+5+6) + 4(7+8+9) = 153
            // 2(1+2) + 0(3) + 3(4+5) + 0(6) + 4(7+8) + 0(9) = 93
            //
            // Row 3:
            // 0(1) + 3(2+3) + 0(4) + 4(5+6) + 0(7) + 5(8+9) = 144
            // 3(1+2+3) + 4(4+5+6) + 5(7+8+9) = 198
            // 3(1+2) + 0(3) + 4(4+5) + 0(6) + 5(7+8) + 0(9) = 120
            //
            // Row 4:
            // 0(1) + 4(2+3) + 0(4) + 5(5+6) + 0(7+8+9) = 75
            // 4(1+2+3) + 5(4+5+6) + 0(7+8+9) = 99
            // 4(1+2) + 0(3) + 5(4+5) + 0(6) + 0(7+8+9) = 57
            std::vector<float> expectedValues{
                45,  63,  63,  63,  39,   // Row 0
                78,  108, 108, 108, 66,   // Row 1
                111, 153, 153, 153, 93,   // Row 2
                144, 198, 198, 198, 120,  // Row 3
                75,  99,  99,  99,  57    // Row 4
            };
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
            weightsTensor->fillData<float>({ 1, 2, 3, 4, 5, 6, 7, 8, 9 });

            convOp->run();

            // Expected output:
            //
            // 1(1+2+3) + 2(4+5+6) + 3(7+8+9) = 108
            // 2(1+2+3) + 3(4+5+6) + 4(7+8+9) = 153
            // 3(1+2+3) + 4(4+5+6) + 5(7+8+9) = 198
            std::vector<float> expectedValues{
                108, 108, 108,  // Row 0
                153, 153, 153,  // Row 1
                198, 198, 198   // Row 2
            };
            auto outputsTensor = convOp->getOutput(0);
            verifyOutputs(outputsTensor, expectedValues);
        }
    }

    SECTION("NHWC layout") {
        TensorShape inputShape({ 1, 5, 5, 2 }, DataLayout::NHWC);
        Tensor* input = new Tensor("input", inputShape);
        input->allocateStorage<float>();
        // Input data looks like:
        input->fillData<float>({
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // Row 0
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  // Row 1
                3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  // Row 2
                4, 4, 4, 4, 4, 4, 4, 4, 4, 4,  // Row 3
                5, 5, 5, 5, 5, 5, 5, 5, 5, 5   // Row 4
        });
        workspace()->addTensor(input);
        convOp->setInput(input, 0);
        SECTION("Same padding, 3x3 kernel, stride 1") {
            convOp->setPadding(SamePadding);
            convOp->setWeightDims(3, 3, 1);
            convOp->setStride(1, 1);
            convOp->createAllTensors();
            allocateAllTensors<float>(convOp);
            // Weight kernel 2D plane:
            // 1  2  3
            // 4  5  6
            // 7  8  9
            auto weightsTensor = convOp->getInput(1);
            weightsTensor->fillData<float>({
                    1, 1, 2, 2, 3, 3,  // Row 0
                    4, 4, 5, 5, 6, 6,  // Row 1
                    7, 7, 8, 8, 9, 9   // Row 2
            });

            convOp->run();

            // Expected output (with zero padding):
            //
            // Row 0:
            // 2 * ( 0(1+2+3) + 0(4) + 1(5+6) + 0(7) + 2(8+9) ) = 90
            // 2 * ( 0(1+2+3) + 1(4+5+6) + 2(7+8+9) ) = 126
            // 2 * ( 0(1+2+3) + 1(4+5) + 0(6) + 2(7+8) + 0(9) ) = 78
            //
            // Row 1:
            // 2 * ( 0(1) + 1(2+3) + 0(4) + 2(5+6) + 0(7) + 3(8+9) ) = 156
            // 2 * ( 1(1+2+3) + 2(4+5+6) + 3(7+8+9) ) = 216
            // 2 * ( 1(1+2) + 0(3) + 2(4+5) + 0(6) + 3(7+8) + 0(9) ) = 132
            //
            // Row 2:
            // 2 * ( 0(1) + 2(2+3) + 0(4) + 3(5+6) + 0(7) + 4(8+9) ) = 222
            // 2 * ( 2(1+2+3) + 3(4+5+6) + 4(7+8+9) ) = 306
            // 2 * ( 2(1+2) + 0(3) + 3(4+5) + 0(6) + 4(7+8) + 0(9) ) = 186
            //
            // Row 3:
            // 2 * ( 0(1) + 3(2+3) + 0(4) + 4(5+6) + 0(7) + 5(8+9) ) = 288
            // 2 * ( 3(1+2+3) + 4(4+5+6) + 5(7+8+9) ) = 396
            // 2 * ( 3(1+2) + 0(3) + 4(4+5) + 0(6) + 5(7+8) + 0(9) ) = 240
            //
            // Row 4:
            // 2 * ( 0(1) + 4(2+3) + 0(4) + 5(5+6) + 0(7+8+9) ) = 150
            // 2 * ( 4(1+2+3) + 5(4+5+6) + 0(7+8+9) ) = 198
            // 2 * ( 4(1+2) + 0(3) + 5(4+5) + 0(6) + 0(7+8+9) ) = 114
            std::vector<float> expectedValues{
                90,  126, 126, 126, 78,   // Row 0
                156, 216, 216, 216, 132,  // Row 1
                222, 306, 306, 306, 186,  // Row 2
                288, 396, 396, 396, 240,  // Row 3
                150, 198, 198, 198, 114   // Row 4
            };
            auto outputsTensor = convOp->getOutput(0);
            verifyOutputs(outputsTensor, expectedValues);
        }

        SECTION("Valid padding, 3x3 kernel, stride 1") {
            convOp->setPadding(ValidPadding);
            convOp->setWeightDims(3, 3, 1);
            convOp->setStride(1, 1);
            convOp->createAllTensors();
            allocateAllTensors<float>(convOp);
            // Weight kernel 2D plane:
            // 1  2  3
            // 4  5  6
            // 7  8  9
            auto weightsTensor = convOp->getInput(1);
            weightsTensor->fillData<float>({
                    1, 1, 2, 2, 3, 3,  // Row 0
                    4, 4, 5, 5, 6, 6,  // Row 1
                    7, 7, 8, 8, 9, 9   // Row 2
            });

            convOp->run();

            // Expected output:
            //
            // 2 * ( 1(1+2+3) + 2(4+5+6) + 3(7+8+9) ) = 216
            // 2 * ( 2(1+2+3) + 3(4+5+6) + 4(7+8+9) ) = 306
            // 2 * ( 3(1+2+3) + 4(4+5+6) + 5(7+8+9) ) = 396
            std::vector<float> expectedValues{
                216, 216, 216,  // Row 0
                306, 306, 306,  // Row 1
                396, 396, 396   // Row 2
            };
            auto outputsTensor = convOp->getOutput(0);
            verifyOutputs(outputsTensor, expectedValues);
        }
    }
}
