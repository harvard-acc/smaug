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
}
