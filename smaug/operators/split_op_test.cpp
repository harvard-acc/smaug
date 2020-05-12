#include "catch.hpp"
#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/core/smaug_test.h"
#include "smaug/operators/split_op.h"

using namespace smaug;

TEST_CASE_METHOD(SmaugTest, "Split a 4D tensor", "[refop]") {
    auto splitOp = new SplitOp<ReferenceBackend>("split", workspace());
    TensorShape inputShape({ 1, 2, 3, 4 }, DataLayout::NCHW);
    Tensor* input = new Tensor("input", inputShape);
    input->allocateStorage<float>();
    std::vector<float> inputValues{
        1,  2,   3,   4,   // input 0, chan 0, row 0
        5,  6,   7,   8,   // input 0, chan 0, row 1
        9,  10,  11,  12,  // input 0, chan 0, row 2
        -1, -2,  -3,  -4,  // input 0, chan 1, row 0
        -5, -6,  -7,  -8,  // input 0, chan 1, row 1
        -9, -10, -11, -12  // input 0, chan 1, row 2
    };
    input->fillData(inputValues.data(), inputValues.size());
    workspace()->addTensor(input);
    splitOp->setInput(input, 0);

    SECTION("Split axis 3, C dimension") {
        splitOp->setSplitAxis(3);
        splitOp->setSplits({ 1, 3 });
        splitOp->createAllTensors();
        allocateAllTensors<float>(splitOp);
        splitOp->run();
        auto output0 = splitOp->getOutput(0);
        auto output1 = splitOp->getOutput(1);
        std::vector<float> expectedValues0{
            1,  5,  9,  // input 0, chan 0
            -1, -5, -9  // input 0, chan 1
        };
        std::vector<float> expectedValues1{
            2,   3,   4,   // input 0, chan 0, row 0
            6,   7,   8,   // input 0, chan 0, row 1
            10,  11,  12,  // input 0, chan 0, row 2
            -2,  -3,  -4,  // input 0, chan 1, row 0
            -6,  -7,  -8,  // input 0, chan 1, row 1
            -10, -11, -12  // input 0, chan 1, row 2
        };
        REQUIRE(output0->getShape().dims() ==
                std::vector<int>{ 1, 2, 3, 1 });
        REQUIRE(output1->getShape().dims() ==
                std::vector<int>{ 1, 2, 3, 3 });
        verifyOutputs(output0, expectedValues0);
        verifyOutputs(output1, expectedValues1);
    }

    SECTION("Split axis 2, H dimension") {
        splitOp->setSplitAxis(2);
        splitOp->setSplits({ 2, 1 });
        splitOp->createAllTensors();
        allocateAllTensors<float>(splitOp);
        splitOp->run();
        auto output0 = splitOp->getOutput(0);
        auto output1 = splitOp->getOutput(1);
        std::vector<float> expectedValues0{
            1,  2,  3,  4,   // input 0, chan 0, row 0
            5,  6,  7,  8,   // input 0, chan 0, row 1
            -1, -2, -3, -4,  // input 0, chan 1, row 0
            -5, -6, -7, -8,  // input 0, chan 1, row 1
        };
        std::vector<float> expectedValues1{
            9,  10,  11,  12,  // input 0, chan 0, row 0
            -9, -10, -11, -12  // input 0, chan 1, row 0
        };
        REQUIRE(output0->getShape().dims() ==
                std::vector<int>{ 1, 2, 2, 4 });
        REQUIRE(output1->getShape().dims() ==
                std::vector<int>{ 1, 2, 1, 4 });
        verifyOutputs(output0, expectedValues0);
        verifyOutputs(output1, expectedValues1);
    }
}

TEST_CASE_METHOD(SmaugTest, "Split a 2D tensor", "[refop]") {
    auto splitOp = new SplitOp<ReferenceBackend>("split", workspace());
    TensorShape inputShape({ 4, 4 }, DataLayout::NC);
    Tensor* input = new Tensor("input", inputShape);
    input->allocateStorage<float>();
    std::vector<float> inputValues{
        1,  2,  3,  4,   // input 0
        5,  6,  7,  8,   // input 1
        9,  10, 11, 12,  // input 2
        13, 14, 15, 16   // input 3
    };
    input->fillData(inputValues.data(), inputValues.size());
    workspace()->addTensor(input);
    splitOp->setInput(input, 0);

    SECTION("Split axis 0, N dimension") {
        splitOp->setSplitAxis(0);
        splitOp->setSplits({ 1, 3 });
        splitOp->createAllTensors();
        allocateAllTensors<float>(splitOp);
        splitOp->run();
        auto output0 = splitOp->getOutput(0);
        auto output1 = splitOp->getOutput(1);
        std::vector<float> expectedValues0{
            1, 2, 3, 4  // input 0
        };
        std::vector<float> expectedValues1{
            5,  6,  7,  8,   // input 0
            9,  10, 11, 12,  // input 1
            13, 14, 15, 16   // input 2
        };
        REQUIRE(output0->getShape().dims() == std::vector<int>{ 1, 4 });
        REQUIRE(output1->getShape().dims() == std::vector<int>{ 3, 4 });
        verifyOutputs(output0, expectedValues0);
        verifyOutputs(output1, expectedValues1);
    }

    SECTION("Split axis 1, C dimension") {
        splitOp->setSplitAxis(1);
        splitOp->setSplits({ 3, 1 });
        splitOp->createAllTensors();
        allocateAllTensors<float>(splitOp);
        splitOp->run();
        auto output0 = splitOp->getOutput(0);
        auto output1 = splitOp->getOutput(1);
        std::vector<float> expectedValues0{
            1,  2,  3,   // input 0
            5,  6,  7,   // input 1
            9,  10, 11,  // input 2
            13, 14, 15   // input 3
        };
        std::vector<float> expectedValues1{
            4, 8, 12, 16  // input 0~3
        };
        REQUIRE(output0->getShape().dims() == std::vector<int>{ 4, 3 });
        REQUIRE(output1->getShape().dims() == std::vector<int>{ 4, 1 });
        verifyOutputs(output0, expectedValues0);
        verifyOutputs(output1, expectedValues1);
    }
}
