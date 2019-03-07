#include "catch.hpp"
#include "core/backend.h"
#include "core/tensor.h"
#include "core/smaug_test.h"
#include "operators/batch_norm_op.h"

using namespace smaug;

TEST_CASE_METHOD(SmaugTest, "Reference batch norm operator", "[refop]") {
    auto bnOp = new BatchNormOp<ReferenceBackend>("batchnorm", workspace());
    SECTION("Batch norm after convolution") {
        TensorShape inputShape({ 1, 4, 4, 4 }, DataLayout::NCHW);
        Tensor* input = new Tensor("input", inputShape);
        input->allocateStorage<float>();
        // Input data looks like:
        input->fillData<float>({ 1, 2, 3, 4,   // chan 1
                                 2, 3, 4, 5,
                                 3, 4, 5, 6,
                                 4, 5, 6, 7,
                                 1, 2, 3, 4,   // chan 2
                                 2, 3, 4, 5,
                                 3, 4, 5, 6,
                                 4, 5, 6, 7,
                                 1, 2, 3, 4,   // chan 3
                                 2, 3, 4, 5,
                                 3, 4, 5, 6,
                                 4, 5, 6, 7,
                                 1, 2, 3, 4,   // chan 4
                                 2, 3, 4, 5,
                                 3, 4, 5, 6,
                                 4, 5, 6, 7 });
        workspace()->addTensor(input);
        bnOp->setInput(input, 0);
        bnOp->createAllTensors();
        allocateAllTensors<float>(bnOp);
        auto meanTensor = bnOp->getInput(1);
        auto varianceTensor = bnOp->getInput(2);
        auto gammaTensor = bnOp->getInput(3);
        auto betaTensor = bnOp->getInput(4);
        meanTensor->fillData<float>({ 1, 2, 3, 4 });
        varianceTensor->fillData<float>({ 5, 6, 7, 8 });
        gammaTensor->fillData<float>({ -1, -2, -3, -4 });
        betaTensor->fillData<float>({ -5, -6, -7, -8 });

        bnOp->run();

        // Expected output:
        //
        // Channel 1:
        // (1 - 1) * (5 * -1) + -5 = -5
        // (2 - 1) * (5 * -1) + -5 = -10
        // (3 - 1) * (5 * -1) + -5 = -15
        // (4 - 1) * (5 * -1) + -5 = -20
        // (5 - 1) * (5 * -1) + -5 = -25
        // (6 - 1) * (5 * -1) + -5 = -30
        // (7 - 1) * (5 * -1) + -5 = -35
        // Channel 2:
        // (1 - 2) * (6 * -2) + -6 = 6
        // (2 - 2) * (6 * -2) + -6 = -6
        // (3 - 2) * (6 * -2) + -6 = -18
        // (4 - 2) * (6 * -2) + -6 = -30
        // (5 - 2) * (6 * -2) + -6 = -42
        // (6 - 2) * (6 * -2) + -6 = -54
        // (7 - 2) * (6 * -2) + -6 = -66
        // Channel 3:
        // (1 - 3) * (7 * -3) + -7 = 35
        // (2 - 3) * (7 * -3) + -7 = 14
        // (3 - 3) * (7 * -3) + -7 = -7
        // (4 - 3) * (7 * -3) + -7 = -28
        // (5 - 3) * (7 * -3) + -7 = -49
        // (6 - 3) * (7 * -3) + -7 = -70
        // (7 - 3) * (7 * -3) + -7 = -91
        // Channel 4:
        // (1 - 4) * (8 * -4) + -8 = 88
        // (2 - 4) * (8 * -4) + -8 = 56
        // (3 - 4) * (8 * -4) + -8 = 24
        // (4 - 4) * (8 * -4) + -8 = -8
        // (5 - 4) * (8 * -4) + -8 = -40
        // (6 - 4) * (8 * -4) + -8 = -72
        // (7 - 4) * (8 * -4) + -8 = -104
        std::vector<float> expectedValues{ -5,-10,-15,-20,  // chan 1
                                           -10,-15,-20,-25,
                                           -15,-20,-25,-30,
                                           -20,-25,-30,-35,
                                           6,-6,-18,-30,  // chan 2
                                           -6,-18,-30,-42,
                                           -18,-30,-42,-54,
                                           -30,-42,-54,-66,
                                           35,14,-7,-28,  // chan 3
                                           14,-7,-28,-49,
                                           -7,-28,-49,-70,
                                           -28,-49,-70,-91,
                                           88,56,24,-8,  // chan 4
                                           56,24,-8,-40,
                                           24,-8,-40,-72,
                                           -8,-40,-72,-104 };
        auto outputsTensor = bnOp->getOutput(0);
        verifyOutputs(outputsTensor, expectedValues);
    }

    SECTION("Batch norm after FC") {
        TensorShape inputShape({ 1, 8 }, DataLayout::NC);
        Tensor* input = new Tensor("input", inputShape);
        input->allocateStorage<float>();
        // Input data looks like:
        input->fillData<float>({ 1, 2, 3, 4, 5, 6, 7, 8 });
        workspace()->addTensor(input);
        bnOp->setInput(input, 0);
        bnOp->createAllTensors();
        allocateAllTensors<float>(bnOp);
        auto meanTensor = bnOp->getInput(1);
        auto varianceTensor = bnOp->getInput(2);
        auto gammaTensor = bnOp->getInput(3);
        auto betaTensor = bnOp->getInput(4);
        meanTensor->fillData<float>({ -1, 1, -2, 2, -3, 3, -4, 4 });
        varianceTensor->fillData<float>({ 0, 1, 2, 3, 0, 1, 2, 3 });
        gammaTensor->fillData<float>({ 3, 2, 1, 0, -1, -2, -3, -4 });
        betaTensor->fillData<float>({ 0, 1, 2, 3, 4, 3, 2, 1 });

        bnOp->run();

        // Expected output:
        //
        // (1 - -1) * (0 *  3) + 0 = 0
        // (2 -  1) * (1 *  2) + 1 = 3
        // (3 - -2) * (2 *  1) + 2 = 12
        // (4 -  2) * (3 *  0) + 3 = 3
        // (5 - -3) * (0 * -1) + 4 = 4
        // (6 -  3) * (1 * -2) + 3 = -3
        // (7 - -4) * (2 * -3) + 2 = -64
        // (8 -  4) * (3 * -4) + 1 = -47
        std::vector<float> expectedValues{ 0, 3, 12, 3, 4, -3, -64, -47 };
        auto outputsTensor = bnOp->getOutput(0);
        verifyOutputs(outputsTensor, expectedValues);
    }
}
