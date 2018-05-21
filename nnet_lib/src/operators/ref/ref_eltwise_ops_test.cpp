#include "catch.hpp"
#include "core/backend.h"
#include "core/tensor.h"
#include "core/smaug_test.h"
#include "operators/eltwise_add_op.h"
#include "operators/relu_op.h"
#include "operators/elu_op.h"
#include "operators/sigmoid_op.h"
#include "operators/tanh_op.h"

using namespace smaug;

TEST_CASE_METHOD(SmaugTest, "Reference eltwise operators", "[refop]") {
    TensorShape inputShape({ 1, 13 }, DataLayout::NC);
    Tensor<ReferenceBackend>* input0 = new Tensor<ReferenceBackend>(
            "input0", inputShape);
    input0->allocateStorage<float>();
    input0->fillData<float>({ -1, -2, -3, 4, 5, 6, 7, 8, 9, -10, 11, -12, 13 });
    workspace()->addTensor(input0);

    SECTION("Element-wise add operator") {
        auto addOp = new EltwiseAddOp<ReferenceBackend>("add", workspace());
        Tensor<ReferenceBackend>* input1 =
                new Tensor<ReferenceBackend>("input1", inputShape);
        input1->allocateStorage<float>();
        input1->fillData<float>(
                { -2, -3, -4, 5, 6, 7, 8, 9, 10, 11, -12, 13, -14 });
        workspace()->addTensor(input1);
        addOp->setInput(input0, 0);
        addOp->setInput(input1, 1);
        addOp->createAllTensors();
        allocateAllTensors<float, ReferenceBackend>(addOp);
        addOp->run();
        std::vector<float> expectedValues{ -3, -5, -7, 9,  11, 13, 15,
                                           17, 19, 1,  -1, 1,  -1 };
        auto outputsTensor = addOp->getOutput<ReferenceBackend>(0);
        verifyOutputs(outputsTensor, expectedValues);
    }
}
