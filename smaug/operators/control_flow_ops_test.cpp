#include "catch.hpp"
#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/core/smaug_test.h"
#include "smaug/operators/control_flow_ops.h"

using namespace smaug;

TEST_CASE_METHOD(SmaugTest, "Control flow operators", "[contrlop]") {
    SECTION("Switch operator") {
        auto switchOp = new SwitchOp<ReferenceBackend>("switch", workspace());
        TensorShape inputShape({ 1, 2, 2, 2 }, DataLayout::NCHW);
        Tensor* input = new Tensor("input", inputShape);
        Tensor* pred = new Tensor("pred", TensorShape({ 1 }, DataLayout::N));
        input->allocateStorage<float>();
        pred->allocateStorage<bool>();
        std::vector<float> inputValues{
            1, 2, 3, 4,  // input 0, chan 0
            5, 6, 7, 8,  // input 0, chan 1
        };
        input->fillData(inputValues.data(), inputValues.size());
        pred->fillData({ true });
        workspace()->addTensor(input);
        workspace()->addTensor(pred);
        switchOp->setInput(input, 0);
        switchOp->setInput(pred, 1);
        switchOp->createAllTensors();
        allocateAllTensors<float>(switchOp);
        switchOp->run();
        auto outputFalse = switchOp->getOutput(0);
        auto outputTrue = switchOp->getOutput(1);
        REQUIRE(outputFalse->isDead() == true);
        REQUIRE(outputTrue->isDead() == false);
        REQUIRE(outputTrue->getShape().dims() == inputShape.dims());
        verifyOutputs<float>(outputTrue, input);
    }

    SECTION("Merge operator") {
        auto mergeOp = new MergeOp<ReferenceBackend>("merge", workspace());
        mergeOp->setNumInputs(3);
        TensorShape inputShape({ 1, 2, 2, 2 }, DataLayout::NCHW);
        Tensor* input0 =
                workspace()->addTensor(new Tensor("input0", inputShape));
        Tensor* input1 =
                workspace()->addTensor(new Tensor("input1", inputShape));
        Tensor* input2 =
                workspace()->addTensor(new Tensor("input2", inputShape));
        input0->allocateStorage<float>();
        input1->allocateStorage<float>();
        input2->allocateStorage<float>();
        std::vector<float> input0Values{
            1, 2, 3, 4, 5, 6, 7, 8,
        };
        std::vector<float> input1Values{
            8, 7, 6, 5, 4, 3, 2, 1,
        };
        std::vector<float> input2Values{
            4, 2, 2, 1, 8, 7, 6, 5,
        };
        input0->fillData(input0Values.data(), input0Values.size());
        input1->fillData(input1Values.data(), input1Values.size());
        input2->fillData(input2Values.data(), input2Values.size());
        // Set all inputs dead except one.
        input0->setDead();
        input1->setDead();
        mergeOp->setInput(input0, 0);
        mergeOp->setInput(input1, 1);
        mergeOp->setInput(input2, 2);
        mergeOp->createAllTensors();
        allocateAllTensors<float>(mergeOp);
        mergeOp->run();
        auto output = mergeOp->getOutput(0);
        REQUIRE(output->getShape().dims() == inputShape.dims());
        verifyOutputs<float>(output, input2);
    }
}
