#include "catch.hpp"
#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/core/smaug_test.h"
#include "smaug/operators/reorder_op.h"

using namespace smaug;

TEST_CASE_METHOD(SmaugTest, "Reorder from NCHW", "[refop]") {
    auto reorderOp = new ReorderOp<ReferenceBackend>("reorder", workspace());
    TensorShape inputShape({ 2, 2, 2, 4 }, DataLayout::NCHW);
    Tensor* input = new Tensor("input", inputShape);
    input->allocateStorage<float>();
    std::vector<float> inputValues{ 1,   2,   3,   4,  // input 0, chan 0
                                    2,   3,   4,   5,
                                    11,  12,  13,  14, // input 0, chan 1
                                    12,  13,  14,  15,
                                    11,  12,  13,  14,  // input 1, chan 0
                                    12,  13,  14,  15,
                                    111, 112, 113, 114, // input 1, chan 1
                                    112, 113, 114, 115 };
    input->fillData(inputValues.data(), inputValues.size());
    workspace()->addTensor(input);
    reorderOp->setInput(input, 0);
    SECTION("To NHWC") {
        reorderOp->setTargetLayout(DataLayout::NHWC);
        reorderOp->createAllTensors();
        allocateAllTensors<float>(reorderOp);
        reorderOp->run();
        auto outputsTensor = reorderOp->getOutput(0);
        std::vector<float> expectedValues{
            1,  11,  2,  12,  3,  13,  4,  14,   // input 0, row 0
            2,  12,  3,  13,  4,  14,  5,  15,   // input 0, row 1
            11, 111, 12, 112, 13, 113, 14, 114,  // input 0, row 0
            12, 112, 13, 113, 14, 114, 15, 115   // input 0, row 1
        };
        REQUIRE(outputsTensor->getShape().getLayout() == DataLayout::NHWC);
        verifyOutputs(outputsTensor, expectedValues);
    }

    SECTION("Flatten") {
        reorderOp->setTargetLayout(DataLayout::NC);
        reorderOp->createAllTensors();
        allocateAllTensors<float>(reorderOp);
        reorderOp->run();
        auto outputsTensor = reorderOp->getOutput(0);
        REQUIRE(outputsTensor->getShape().getLayout() == DataLayout::NC);
        verifyOutputs(outputsTensor, inputValues);
    }
}

TEST_CASE_METHOD(SmaugTest, "Reorder from NHWC", "[refop]") {
    auto reorderOp = new ReorderOp<ReferenceBackend>("reorder", workspace());
    TensorShape inputShape({ 2, 2, 4, 2 }, DataLayout::NHWC);
    Tensor* input = new Tensor("input", inputShape);
    input->allocateStorage<float>();
    std::vector<float> inputValues{
        1,  11,  2,  12,  3,  13,  4,  14,   // input 0, row 0
        2,  12,  3,  13,  4,  14,  5,  15,   // input 0, row 1
        11, 111, 12, 112, 13, 113, 14, 114,  // input 0, row 0
        12, 112, 13, 113, 14, 114, 15, 115   // input 0, row 1
    };
    input->fillData(inputValues.data(), inputValues.size());
    workspace()->addTensor(input);
    reorderOp->setInput(input, 0);
    SECTION("To NCHW") {
        reorderOp->setTargetLayout(DataLayout::NCHW);
        reorderOp->createAllTensors();
        allocateAllTensors<float>(reorderOp);
        reorderOp->run();
        auto outputsTensor = reorderOp->getOutput(0);
        std::vector<float> expectedValues{
            1,   2,   3,   4,   // input 0, chan 0
            2,   3,   4,   5,
            11,  12,  13,  14,  // input 0, chan 1
            12,  13,  14,  15,
            11,  12,  13,  14,  // input 1, chan 0
            12,  13,  14,  15,
            111, 112, 113, 114, // input 1, chan 1
            112, 113, 114, 115
        };
        REQUIRE(outputsTensor->getShape().getLayout() == DataLayout::NCHW);
        verifyOutputs(outputsTensor, expectedValues);
    }

    SECTION("Flatten") {
        reorderOp->setTargetLayout(DataLayout::NC);
        reorderOp->createAllTensors();
        allocateAllTensors<float>(reorderOp);
        reorderOp->run();
        auto outputsTensor = reorderOp->getOutput(0);
        REQUIRE(outputsTensor->getShape().getLayout() == DataLayout::NC);
        verifyOutputs(outputsTensor, inputValues);
    }
}
