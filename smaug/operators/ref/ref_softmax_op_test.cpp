#include "catch.hpp"
#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/core/smaug_test.h"
#include "smaug/operators/softmax_op.h"

using namespace smaug;

TEST_CASE_METHOD(SmaugTest, "Reference softmax operator", "[refop]") {
    TensorShape inputShape({ 1, 10 }, DataLayout::NC);
    Tensor* input = new Tensor("input", inputShape);
    input->allocateStorage<float>();
    input->fillData<float>({ -10, -8, -6, -4, -2, 0, 2, 4, 6, 8 });
    workspace()->addTensor(input);

    SECTION("Softmax operator") {
        auto softmaxOp =
                new SoftmaxOp<ReferenceBackend>("softmax", workspace());
        softmaxOp->setInput(input, 0);
        softmaxOp->createAllTensors();
        allocateAllTensors<float>(softmaxOp);
        softmaxOp->run();
        std::vector<float> expectedValues{
            1.316882e-8, 9.730519e-8, 7.189935e-7, 5.312683e-6, 3.925571e-5,
            2.900626e-4, 0.002143289, 0.015836887, 0.117019645, 0.864664718
        };
        auto outputsTensor = softmaxOp->getOutput(0);
        verifyOutputs(outputsTensor, expectedValues);
    }
}
