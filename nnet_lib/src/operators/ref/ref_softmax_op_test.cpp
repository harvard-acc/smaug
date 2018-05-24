#include "catch.hpp"
#include "core/backend.h"
#include "core/tensor.h"
#include "core/smaug_test.h"
#include "operators/softmax_op.h"

using namespace smaug;

TEST_CASE_METHOD(SmaugTest, "Reference softmax operator", "[refop]") {
    TensorShape inputShape({ 1, 10 }, DataLayout::NC);
    Tensor<ReferenceBackend>* input = new Tensor<ReferenceBackend>(
            "input", inputShape);
    input->allocateStorage<float>();
    input->fillData<float>({ -10, -8, -6, -4, -2, 0, 2, 4, 6, 8 });
    workspace()->addTensor(input);

    SECTION("Softmax operator") {
        auto softmaxOp =
                new SoftmaxOp<ReferenceBackend>("softmax", workspace());
        softmaxOp->setInput(input, 0);
        softmaxOp->createAllTensors();
        allocateAllTensors<float, ReferenceBackend>(softmaxOp);
        softmaxOp->run();
        std::vector<float> expectedValues{
            1.316882e-8, 9.730519e-8, 7.189935e-7, 5.312683e-6, 3.925571e-5,
            2.900626e-4, 0.002143289, 0.015836887, 0.117019645, 0.864664718
        };
        auto outputsTensor = softmaxOp->getOutput<ReferenceBackend>(0);
        verifyOutputs(outputsTensor, expectedValues);
    }
}
