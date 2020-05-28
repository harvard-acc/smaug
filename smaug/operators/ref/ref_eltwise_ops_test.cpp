#include "catch.hpp"
#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/core/smaug_test.h"
#include "smaug/operators/eltwise_add_op.h"
#include "smaug/operators/eltwise_mul_op.h"
#include "smaug/operators/less_op.h"
#include "smaug/operators/greater_op.h"
#include "smaug/operators/relu_op.h"
#include "smaug/operators/elu_op.h"
#include "smaug/operators/sigmoid_op.h"
#include "smaug/operators/tanh_op.h"

using namespace smaug;

TEST_CASE_METHOD(SmaugTest, "Reference eltwise operators", "[refop]") {
    TensorShape inputShape({ 1, 13 }, DataLayout::NC);
    Tensor* input0 = new Tensor("input0", inputShape);
    input0->allocateStorage<float>();
    input0->fillData<float>({ -1, -2, -3, 4, 5, 6, 7, 8, 9, -10, 11, -12, 13 });
    workspace()->addTensor(input0);

    SECTION("Element-wise add operator") {
        auto addOp = new EltwiseAddOp<ReferenceBackend>("add", workspace());
        Tensor* input1 = new Tensor("input1", inputShape);
        input1->allocateStorage<float>();
        input1->fillData<float>(
                { -2, -3, -4, 5, 6, 7, 8, 9, 10, 11, -12, 13, -14 });
        workspace()->addTensor(input1);
        addOp->setInput(input0, 0);
        addOp->setInput(input1, 1);
        addOp->createAllTensors();
        allocateAllTensors<float>(addOp);
        addOp->run();
        std::vector<float> expectedValues{ -3, -5, -7, 9,  11, 13, 15,
                                           17, 19, 1,  -1, 1,  -1 };
        auto outputsTensor = addOp->getOutput(0);
        verifyOutputs(outputsTensor, expectedValues);
    }

    SECTION("Element-wise mul operator") {
        auto mulOp = new EltwiseMulOp<ReferenceBackend>("mul", workspace());
        Tensor* input1 = new Tensor("input1", inputShape);
        input1->allocateStorage<float>();
        input1->fillData<float>(
                { -2, -3, -4, 5, 6, 7, 8, 9, 10, 11, -12, 13, -14 });
        workspace()->addTensor(input1);
        mulOp->setInput(input0, 0);
        mulOp->setInput(input1, 1);
        mulOp->createAllTensors();
        allocateAllTensors<float>(mulOp);
        mulOp->run();
        std::vector<float> expectedValues{ 2, 6, 12, 20, 30, 42, 56,
                                           72, 90, -110, -132, -156, -182 };
        auto outputsTensor = mulOp->getOutput(0);
        verifyOutputs(outputsTensor, expectedValues);
    }

    SECTION("Less operator") {
        TensorShape inputShape({ 1, 4 }, DataLayout::NC);
        auto lessOp = new LessOp<ReferenceBackend>("less", workspace());
        Tensor* input0 = new Tensor("input0", inputShape);
        input0->allocateStorage<float>();
        input0->fillData<float>({ -2, 3, -4, 5 });
        Tensor* input1 = new Tensor("input1", inputShape);
        input1->allocateStorage<float>();
        input1->fillData<float>({ -2, 7, -8, 9 });
        workspace()->addTensor(input1);
        lessOp->setInput(input0, 0);
        lessOp->setInput(input1, 1);
        lessOp->createAllTensors();
        lessOp->getOutput(0)->allocateStorage<bool>();
        lessOp->run();
        std::vector<bool> expectedValues{ false, true, false, true };
        auto outputsTensor = lessOp->getOutput(0);
        verifyOutputs(outputsTensor, expectedValues);
    }

    SECTION("LessEqual operator") {
        TensorShape inputShape({ 1, 4 }, DataLayout::NC);
        auto lessEqualOp =
                new LessEqualOp<ReferenceBackend>("lessEqual", workspace());
        Tensor* input0 = new Tensor("input0", inputShape);
        input0->allocateStorage<float>();
        input0->fillData<float>({ -2, 3, -4, 5 });
        Tensor* input1 = new Tensor("input1", inputShape);
        input1->allocateStorage<float>();
        input1->fillData<float>({ -2, 7, -8, 9 });
        workspace()->addTensor(input1);
        lessEqualOp->setInput(input0, 0);
        lessEqualOp->setInput(input1, 1);
        lessEqualOp->createAllTensors();
        lessEqualOp->getOutput(0)->allocateStorage<bool>();
        lessEqualOp->run();
        std::vector<bool> expectedValues{ true, true, false, true };
        auto outputsTensor = lessEqualOp->getOutput(0);
        verifyOutputs(outputsTensor, expectedValues);
    }

    SECTION("Greater operator") {
        TensorShape inputShape({ 1, 4 }, DataLayout::NC);
        auto greaterOp =
                new GreaterOp<ReferenceBackend>("greater", workspace());
        Tensor* input0 = new Tensor("input0", inputShape);
        input0->allocateStorage<float>();
        input0->fillData<float>({ -2, 3, -4, 5 });
        Tensor* input1 = new Tensor("input1", inputShape);
        input1->allocateStorage<float>();
        input1->fillData<float>({ -2, 7, -8, 9 });
        workspace()->addTensor(input1);
        greaterOp->setInput(input0, 0);
        greaterOp->setInput(input1, 1);
        greaterOp->createAllTensors();
        greaterOp->getOutput(0)->allocateStorage<bool>();
        greaterOp->run();
        std::vector<bool> expectedValues{ false, false, true, false };
        auto outputsTensor = greaterOp->getOutput(0);
        verifyOutputs(outputsTensor, expectedValues);
    }

    SECTION("GreaterEqual operator") {
        TensorShape inputShape({ 1, 4 }, DataLayout::NC);
        auto greaterEqualOp = new GreaterEqualOp<ReferenceBackend>(
                "greaterEqual", workspace());
        Tensor* input0 = new Tensor("input0", inputShape);
        input0->allocateStorage<float>();
        input0->fillData<float>({ -2, 3, -4, 5 });
        Tensor* input1 = new Tensor("input1", inputShape);
        input1->allocateStorage<float>();
        input1->fillData<float>({ -2, 7, -8, 9 });
        workspace()->addTensor(input1);
        greaterEqualOp->setInput(input0, 0);
        greaterEqualOp->setInput(input1, 1);
        greaterEqualOp->createAllTensors();
        greaterEqualOp->getOutput(0)->allocateStorage<bool>();
        greaterEqualOp->run();
        std::vector<bool> expectedValues{ true, false, true, false };
        auto outputsTensor = greaterEqualOp->getOutput(0);
        verifyOutputs(outputsTensor, expectedValues);
    }

    SECTION("RELU") {
        auto reluOp = new ReluOp<ReferenceBackend>("relu", workspace());
        reluOp->setInput(input0, 0);

        SECTION("Slope 0") {
            reluOp->createAllTensors();
            allocateAllTensors<float>(reluOp);
            reluOp->run();
            std::vector<float> expectedValues{ 0, 0, 0, 4, 5, 6, 7,
                                               8, 9, 0, 11, 0, 13 };
            auto outputsTensor = reluOp->getOutput(0);
            verifyOutputs(outputsTensor, expectedValues);
        }

        SECTION("Slope 0.1") {
            reluOp->setSlope(0.1);
            reluOp->createAllTensors();
            allocateAllTensors<float>(reluOp);
            reluOp->run();
            std::vector<float> expectedValues{
                -0.1, -0.2, -0.3, 4, 5, 6, 7, 8, 9, -1, 11, -1.2, 13
            };
            auto outputsTensor = reluOp->getOutput(0);
            verifyOutputs(outputsTensor, expectedValues);
        }
    }

    SECTION("ELU") {
        auto eluOp = new EluOp<ReferenceBackend>("elu", workspace(), 0.1);
        eluOp->setInput(input0, 0);
        eluOp->createAllTensors();
        allocateAllTensors<float>(eluOp);
        eluOp->run();
        std::vector<float> expectedValues{
            -0.063212, -0.086466, -0.0950213, 4,  5,           6, 7,
            8,         9,         -0.099995,  11, -0.09999939, 13
        };
        auto outputsTensor = eluOp->getOutput(0);
        verifyOutputs(outputsTensor, expectedValues);
    }

    SECTION("SELU") {
        auto seluOp = new SeluOp<ReferenceBackend>("selu", workspace());
        seluOp->setInput(input0, 0);
        seluOp->createAllTensors();
        allocateAllTensors<float>(seluOp);
        seluOp->run();
        std::vector<float> expectedValues{
            -1.111354, -1.520198, -1.6706,   4.2028,  5.2535,    6.3042, 7.3549,
            8.4056,    9.4563,    -1.758056, 11.5577, -1.758126, 13.6591
        };
        auto outputsTensor = seluOp->getOutput(0);
        verifyOutputs(outputsTensor, expectedValues);
    }
}

TEST_CASE_METHOD(SmaugTest, "Reference saturating nonlinearities", "[refop]") {
    TensorShape inputShape({ 1, 11 }, DataLayout::NC);
    Tensor* input0 = new Tensor("input0", inputShape);
    input0->allocateStorage<float>();
    input0->fillData<float>(
            { -1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1 });
    workspace()->addTensor(input0);
    SECTION("Sigmoid") {
        auto sigmoidOp = new SigmoidOp<ReferenceBackend>("sigmoid", workspace());
        sigmoidOp->setInput(input0, 0);
        sigmoidOp->createAllTensors();
        allocateAllTensors<float>(sigmoidOp);
        sigmoidOp->run();

        std::vector<float> expectedValues{ 0.2689414,  0.3100255, 0.354344,
                                           0.40131234, 0.4501660, 0.5,
                                           0.549834,   0.5986876, 0.6456563,
                                           0.6899744,  0.7310586 };
        auto outputsTensor = sigmoidOp->getOutput(0);
        verifyOutputs(outputsTensor, expectedValues);
    }

    SECTION("Tanh") {
        auto tanhOp = new TanhOp<ReferenceBackend>("tanh", workspace());
        tanhOp->setInput(input0, 0);
        tanhOp->createAllTensors();
        allocateAllTensors<float>(tanhOp);
        tanhOp->run();

        std::vector<float> expectedValues{ -0.761594, -0.6640367,   -0.5370496,
                                           -0.379949, -0.1973753, 0,
                                           0.1973753, 0.379949,   0.5370496,
                                           0.6640367, 0.761594 };
        auto outputsTensor = tanhOp->getOutput(0);
        verifyOutputs(outputsTensor, expectedValues);
    }

    SECTION("Hard Tanh") {
        auto hardTanhOp = new HardTanhOp<ReferenceBackend>(
                "hardTanh", workspace(), -0.5, 0.5);
        hardTanhOp->setInput(input0, 0);
        hardTanhOp->createAllTensors();
        allocateAllTensors<float>(hardTanhOp);
        hardTanhOp->run();

        std::vector<float> expectedValues{ -0.5, -0.5, -0.5, -0.4, -0.2, 0,
                                           0.2,  0.4,  0.5,  0.5,  0.5 };
        auto outputsTensor = hardTanhOp->getOutput(0);
        verifyOutputs(outputsTensor, expectedValues);
    }
}
