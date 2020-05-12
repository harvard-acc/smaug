#include "catch.hpp"
#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/core/smaug_test.h"
#include "smaug/operators/repeat_op.h"

using namespace smaug;

TEST_CASE_METHOD(SmaugTest, "Repeat dims of a tensor", "[refop]") {
    auto repeatOp = new RepeatOp<ReferenceBackend>("repeat", workspace());
    TensorShape inputShape({ 1, 2, 2, 2 }, DataLayout::NCHW);
    Tensor* input = new Tensor("input", inputShape);
    input->allocateStorage<float>();
    std::vector<float> inputValues{
        1, 2, 3, 4,  // input 0, chan 0
        5, 6, 7, 8,  // input 0, chan 1
    };
    input->fillData(inputValues.data(), inputValues.size());
    workspace()->addTensor(input);
    repeatOp->setInput(input, 0);

    SECTION("Repeat 1 dimension") {
        SECTION("Repeat axis 3") {
            repeatOp->setMultiples({ 1, 1, 1, 3 });
            repeatOp->createAllTensors();
            allocateAllTensors<float>(repeatOp);
            repeatOp->run();
            auto output = repeatOp->getOutput(0);
            std::vector<float> expectedValues{
                1, 2, 1, 2, 1, 2,  // input 0, chan 0, row 0
                3, 4, 3, 4, 3, 4,  // input 0, chan 0, row 1
                5, 6, 5, 6, 5, 6,  // input 0, chan 1, row 0
                7, 8, 7, 8, 7, 8   // input 0, chan 1, row 1
            };
            REQUIRE(output->getShape().dims() ==
                    std::vector<int>{ 1, 2, 2, 6 });
            verifyOutputs(output, expectedValues);
        }
        SECTION("Repeat axis 2") {
            repeatOp->setMultiples({ 1, 1, 3, 1 });
            repeatOp->createAllTensors();
            allocateAllTensors<float>(repeatOp);
            repeatOp->run();
            auto output = repeatOp->getOutput(0);
            std::vector<float> expectedValues{
                1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,  // input 0, chan 0
                5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8   // input 0, chan 1
            };
            REQUIRE(output->getShape().dims() ==
                    std::vector<int>{ 1, 2, 6, 2 });
            verifyOutputs(output, expectedValues);
        }
        SECTION("Repeat axis 0") {
            repeatOp->setMultiples({ 3, 1, 1, 1 });
            repeatOp->createAllTensors();
            allocateAllTensors<float>(repeatOp);
            repeatOp->run();
            auto output = repeatOp->getOutput(0);
            std::vector<float> expectedValues{
                1, 2, 3, 4,  // input 0, chan 0
                5, 6, 7, 8,  // input 0, chan 1
                1, 2, 3, 4,  // input 1, chan 0
                5, 6, 7, 8,  // input 1, chan 1
                1, 2, 3, 4,  // input 2, chan 0
                5, 6, 7, 8,  // input 2, chan 1
            };
            REQUIRE(output->getShape().dims() ==
                    std::vector<int>{ 3, 2, 2, 2 });
            verifyOutputs(output, expectedValues);
        }
    }

    SECTION("Repeat multiple dimensions") {
        SECTION("Repeat axes 2 and 3") {
            repeatOp->setMultiples({ 1, 1, 2, 3 });
            repeatOp->createAllTensors();
            allocateAllTensors<float>(repeatOp);
            repeatOp->run();
            auto output = repeatOp->getOutput(0);
            std::vector<float> expectedValues{
                1, 2, 1, 2, 1, 2,  // input 0, chan 0, row 0
                3, 4, 3, 4, 3, 4,  // input 0, chan 0, row 1
                1, 2, 1, 2, 1, 2,  // input 0, chan 0, row 2
                3, 4, 3, 4, 3, 4,  // input 0, chan 0, row 3
                5, 6, 5, 6, 5, 6,  // input 0, chan 1, row 0
                7, 8, 7, 8, 7, 8,  // input 0, chan 1, row 1
                5, 6, 5, 6, 5, 6,  // input 0, chan 1, row 2
                7, 8, 7, 8, 7, 8   // input 0, chan 1, row 3
            };
            REQUIRE(output->getShape().dims() ==
                    std::vector<int>{ 1, 2, 4, 6 });
            verifyOutputs(output, expectedValues);
        }
        SECTION("Repeat axes 0, 1 and 3") {
            repeatOp->setMultiples({ 3, 2, 1, 3 });
            repeatOp->createAllTensors();
            allocateAllTensors<float>(repeatOp);
            repeatOp->run();
            auto output = repeatOp->getOutput(0);
            std::vector<float> expectedValues{
                1, 2, 1, 2, 1, 2,  // input 0, chan 0, row 0
                3, 4, 3, 4, 3, 4,  // input 0, chan 0, row 1
                5, 6, 5, 6, 5, 6,  // input 0, chan 0, row 0
                7, 8, 7, 8, 7, 8,  // input 0, chan 0, row 1
                1, 2, 1, 2, 1, 2,  // input 0, chan 1, row 0
                3, 4, 3, 4, 3, 4,  // input 0, chan 1, row 1
                5, 6, 5, 6, 5, 6,  // input 0, chan 1, row 0
                7, 8, 7, 8, 7, 8,  // input 0, chan 1, row 1
                1, 2, 1, 2, 1, 2,  // input 1, chan 0, row 0
                3, 4, 3, 4, 3, 4,  // input 1, chan 0, row 1
                5, 6, 5, 6, 5, 6,  // input 1, chan 0, row 0
                7, 8, 7, 8, 7, 8,  // input 1, chan 0, row 1
                1, 2, 1, 2, 1, 2,  // input 1, chan 1, row 0
                3, 4, 3, 4, 3, 4,  // input 1, chan 1, row 1
                5, 6, 5, 6, 5, 6,  // input 1, chan 1, row 0
                7, 8, 7, 8, 7, 8,  // input 1, chan 1, row 1
                1, 2, 1, 2, 1, 2,  // input 2, chan 0, row 0
                3, 4, 3, 4, 3, 4,  // input 2, chan 0, row 1
                5, 6, 5, 6, 5, 6,  // input 2, chan 0, row 0
                7, 8, 7, 8, 7, 8,  // input 2, chan 0, row 1
                1, 2, 1, 2, 1, 2,  // input 2, chan 1, row 0
                3, 4, 3, 4, 3, 4,  // input 2, chan 1, row 1
                5, 6, 5, 6, 5, 6,  // input 2, chan 1, row 0
                7, 8, 7, 8, 7, 8   // input 2, chan 1, row 1
            };
            REQUIRE(output->getShape().dims() ==
                    std::vector<int>{ 3, 4, 2, 6 });
            verifyOutputs(output, expectedValues);
        }
    }
}
