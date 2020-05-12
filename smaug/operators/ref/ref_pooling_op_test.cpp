#include "catch.hpp"
#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/core/smaug_test.h"
#include "smaug/operators/pooling_op.h"

using namespace smaug;

TEST_CASE_METHOD(SmaugTest,
                 "Reference pooling operators",
                 "[refop]") {
    SECTION("2x2 pooling") {
        TensorShape inputShape({ 1, 2, 10, 10 }, DataLayout::NCHW);
        Tensor* input = new Tensor("input", inputShape);
        input->allocateStorage<float>();
        input->fillData<float>({ 1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                 2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                 3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                 4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                 5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                 6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                 7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                 8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                                 9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                                 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                 -1, -2, -3,  -4,  -5,  -6,  -7,  -8,  -9,  -10,
                                 -2, -3, -4,  -5,  -6,  -7,  -8,  -9,  -10, -11,
                                 -3, -4, -5,  -6,  -7,  -8,  -9,  -10, -11, -12,
                                 -4, -5, -6,  -7,  -8,  -9,  -10, -11, -12, -13,
                                 -5, -6, -7,  -8,  -9,  -10, -11, -12, -13, -14,
                                 -6, -7, -8,  -9,  -10, -11, -12, -13, -14, -15,
                                 -7, -8, -9,  -10, -11, -12, -13, -14, -15, -16,
                                 -8, -9, -10, -11, -12, -13, -14, -15, -16, -17,
                                 -9, -10,-11, -12, -13, -14, -15, -16, -17, -18,
                                 -10,-11,-12, -13, -14, -15, -16, -17, -18, -19
                                 });
        workspace()->addTensor(input);
        SECTION("max pooling with stride 2") {
            auto poolOp =
                    new MaxPoolingOp<ReferenceBackend>("pool", workspace());
            poolOp->setInput(input, 0);
            poolOp->setPoolingSize(2, 2);
            poolOp->setPoolingStride(2, 2);
            poolOp->createAllTensors();
            allocateAllTensors<float>(poolOp);
            poolOp->run();

            std::vector<float> expectedValues{ 3,  5,  7,  9,  11,
                                               5,  7,  9,  11, 13,
                                               7,  9,  11, 13, 15,
                                               9,  11, 13, 15, 17,
                                               11, 13, 15, 17, 19,
                                               -1, -3,  -5,  -7,  -9,
                                               -3, -5,  -7,  -9,  -11,
                                               -5, -7,  -9,  -11, -13,
                                               -7, -9,  -11, -13, -15,
                                               -9, -11, -13, -15, -17 };
            auto outputsTensor = poolOp->getOutput(0);
            verifyOutputs(outputsTensor, expectedValues);
        }

        SECTION("average pooling with stride 2") {
            auto poolOp =
                    new AvgPoolingOp<ReferenceBackend>("pool", workspace());
            poolOp->setInput(input, 0);
            poolOp->setPoolingSize(2, 2);
            poolOp->setPoolingStride(2, 2);
            poolOp->createAllTensors();
            allocateAllTensors<float>(poolOp);
            poolOp->run();

            std::vector<float> expectedValues{ 2,  4,  6,  8,  10,
                                               4,  6,  8,  10, 12,
                                               6,  8,  10, 12, 14,
                                               8,  10, 12, 14, 16,
                                               10, 12, 14, 16, 18,
                                               -2, -4, -6, -8, -10,
                                               -4, -6, -8, -10,-12,
                                               -6, -8, -10,-12,-14,
                                               -8, -10,-12,-14,-16,
                                               -10,-12,-14,-16,-18 };
            auto outputsTensor = poolOp->getOutput(0);
            verifyOutputs(outputsTensor, expectedValues);
        }
    }

    SECTION("3x3 pooling") {
        TensorShape inputShape({ 1, 2, 11, 11 }, DataLayout::NCHW);
        Tensor* input = new Tensor("input", inputShape);
        input->allocateStorage<float>();
        input->fillData<float>({ 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                 2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                 3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                                 4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                 5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                 6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                 7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                                 8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                                 9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                            -1, -2, -3,  -4,  -5,  -6,  -7,  -8,  -9,  -10, -11,
                            -2, -3, -4,  -5,  -6,  -7,  -8,  -9,  -10, -11, -12,
                            -3, -4, -5,  -6,  -7,  -8,  -9,  -10, -11, -12, -13,
                            -4, -5, -6,  -7,  -8,  -9,  -10, -11, -12, -13, -14,
                            -5, -6, -7,  -8,  -9,  -10, -11, -12, -13, -14, -15,
                            -6, -7, -8,  -9,  -10, -11, -12, -13, -14, -15, -16,
                            -7, -8, -9,  -10, -11, -12, -13, -14, -15, -16, -17,
                            -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18,
                            -9, -10,-11, -12, -13, -14, -15, -16, -17, -18, -19,
                            -10,-11,-12, -13, -14, -15, -16, -17, -18, -19, -20,
                            -11,-12,-13, -14, -15, -16, -17, -18, -19, -20, -21
                                 });
        workspace()->addTensor(input);
        SECTION("max pooling with stride 2") {
            auto poolOp =
                    new MaxPoolingOp<ReferenceBackend>("pool", workspace());
            poolOp->setInput(input, 0);
            poolOp->setPoolingSize(3, 3);
            poolOp->setPoolingStride(2, 2);
            poolOp->createAllTensors();
            allocateAllTensors<float>(poolOp);
            poolOp->run();

            std::vector<float> expectedValues{ 5,  7,  9,  11, 13,
                                               7,  9,  11, 13, 15,
                                               9,  11, 13, 15, 17,
                                               11, 13, 15, 17, 19,
                                               13, 15, 17, 19, 21,
                                           -1,  -3,  -5,  -7,  -9,
                                           -3,  -5,  -7,  -9, -11,
                                           -5,  -7,  -9, -11, -13,
                                           -7,  -9, -11, -13, -15,
                                           -9, -11, -13, -15, -17 };
            auto outputsTensor = poolOp->getOutput(0);
            verifyOutputs(outputsTensor, expectedValues);
        }

        SECTION("average pooling with stride 2") {
            auto poolOp =
                    new AvgPoolingOp<ReferenceBackend>("pool", workspace());
            poolOp->setInput(input, 0);
            poolOp->setPoolingSize(3, 3);
            poolOp->setPoolingStride(2, 2);
            poolOp->createAllTensors();
            allocateAllTensors<float>(poolOp);
            poolOp->run();

            std::vector<float> expectedValues{ 3,  5,  7,  9,  11,
                                               5,  7,  9,  11, 13,
                                               7,  9,  11, 13, 15,
                                               9,  11, 13, 15, 17,
                                               11, 13, 15, 17, 19,
                                           -3,  -5,  -7,  -9, -11,
                                           -5,  -7,  -9, -11, -13,
                                           -7,  -9, -11, -13, -15,
                                           -9, -11, -13, -15, -17,
                                          -11, -13, -15, -17, -19 };
            auto outputsTensor = poolOp->getOutput(0);
            verifyOutputs(outputsTensor, expectedValues);
        }
    }
}
