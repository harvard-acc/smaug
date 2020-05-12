#include "catch.hpp"
#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/core/smaug_test.h"
#include "smaug/operators/concat_op.h"

using namespace smaug;

TEST_CASE_METHOD(SmaugTest, "Concatenate 2 tensors", "[refop]") {
    auto concatOp =
            new ConcatOp<ReferenceBackend>("concat", workspace(), 2);
    SECTION("Tensors of the same shape") {
        TensorShape inputShape({ 2, 2, 2, 2 }, DataLayout::NCHW);
        Tensor* input0 = new Tensor("input0", inputShape);
        Tensor* input1 = new Tensor("input1", inputShape);
        input0->allocateStorage<float>();
        input1->allocateStorage<float>();
        std::vector<float> inputValues0{
            1,  2,  3,  4,   // input 0, chan 0
            5,  6,  7,  8,   // input 0, chan 1
            9,  10, 11, 12,  // input 1, chan 0
            13, 14, 15, 16   // input 1, chan 1
        };
        std::vector<float> inputValues1(inputValues0.size());
        for (int i = 0; i < inputValues1.size(); i++)
            inputValues1[i] = inputValues0[i] + 20;
        input0->fillData(inputValues0.data(), inputValues0.size());
        input1->fillData(inputValues1.data(), inputValues1.size());
        workspace()->addTensor(input0);
        workspace()->addTensor(input1);
        concatOp->setInput(input0, 0);
        concatOp->setInput(input1, 1);
        SECTION("Concatenate axis 3, W dimension") {
            concatOp->setConcatAxis(3);
            concatOp->createAllTensors();
            allocateAllTensors<float>(concatOp);
            concatOp->run();
            auto outputsTensor = concatOp->getOutput(0);
            std::vector<float> expectedValues{
                1,  2,  21, 22, 3,  4,  23, 24,  // input 0, chan 0
                5,  6,  25, 26, 7,  8,  27, 28,  // input 0, chan 1
                9,  10, 29, 30, 11, 12, 31, 32,  // input 1, chan 0
                13, 14, 33, 34, 15, 16, 35, 36   // input 1, chan 1
            };
            REQUIRE(outputsTensor->getShape().dims() ==
                    std::vector<int>{ 2, 2, 2, 4 });
            verifyOutputs(outputsTensor, expectedValues);
        }

        SECTION("Concatenate axis 1, C dimension") {
            concatOp->setConcatAxis(1);
            concatOp->createAllTensors();
            allocateAllTensors<float>(concatOp);
            concatOp->run();
            auto outputsTensor = concatOp->getOutput(0);
            std::vector<float> expectedValues{
                1,  2,  3,  4,   // input 0, chan 0
                5,  6,  7,  8,   // input 0, chan 1
                21, 22, 23, 24,  // input 0, chan 2
                25, 26, 27, 28,  // input 0, chan 3
                9,  10, 11, 12,  // input 1, chan 0
                13, 14, 15, 16,  // input 1, chan 1
                29, 30, 31, 32,  // input 1, chan 2
                33, 34, 35, 36   // input 1, chan 3
            };
            REQUIRE(outputsTensor->getShape().dims() ==
                    std::vector<int>{ 2, 4, 2, 2 });
            verifyOutputs(outputsTensor, expectedValues);
        }
    }
    SECTION("Tensors of different shapes") {
        TensorShape inputShape0({ 1, 3, 2, 2 }, DataLayout::NCHW);
        TensorShape inputShape1({ 1, 3, 3, 2 }, DataLayout::NCHW);
        Tensor* input0 = new Tensor("input0", inputShape0);
        Tensor* input1 = new Tensor("input1", inputShape1);
        input0->allocateStorage<float>();
        input1->allocateStorage<float>();
        std::vector<float> input0Values{
            1, 2,  3,  4,   // chan 0
            5, 6,  7,  8,   // chan 1
            9, 10, 11, 12,  // chan 2
        };
        std::vector<float> input1Values{
            1,  2,  3,  4,  5,  6,   // chan 0
            7,  8,  9,  10, 11, 12,  // chan 1
            13, 14, 15, 16, 17, 18   // chan 2
        };
        input0->fillData(input0Values.data(), input0Values.size());
        input1->fillData(input1Values.data(), input1Values.size());
        workspace()->addTensor(input0);
        workspace()->addTensor(input1);
        concatOp->setInput(input0, 0);
        concatOp->setInput(input1, 1);
        concatOp->setConcatAxis(2);
        concatOp->createAllTensors();
        allocateAllTensors<float>(concatOp);
        concatOp->run();
        auto outputsTensor = concatOp->getOutput(0);
        std::vector<float> expectedValues{
            1, 2,  3,  4,  1,  2,  3,  4,  5,  6,   // chan 0
            5, 6,  7,  8,  7,  8,  9,  10, 11, 12,  // chan 1
            9, 10, 11, 12, 13, 14, 15, 16, 17, 18   // chan 2
        };
        REQUIRE(outputsTensor->getShape().dims() ==
                std::vector<int>{ 1, 3, 5, 2 });
        verifyOutputs(outputsTensor, expectedValues);
    }
}
