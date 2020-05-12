#include "catch.hpp"
#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/core/smaug_test.h"
#include "smaug/operators/reorder_op.h"
#include "smaug/operators/inner_product_op.h"

using namespace smaug;

Tensor* transposeWeights(Tensor* weights, Workspace* workspace) {
    auto weightsTransOp =
            new ReorderOp<ReferenceBackend>("weights/trans", workspace);
    weightsTransOp->setTargetLayout(NC);
    weightsTransOp->setInput(weights, 0);
    weightsTransOp->createAllTensors();
    weightsTransOp->getOutput(0)->allocateStorage<float>();
    weightsTransOp->run();
    return weightsTransOp->getOutput(0);
}

TEST_CASE_METHOD(SmaugTest, "Reference inner product operator", "[refop]") {
    auto matMulOp = new InnerProductOp<ReferenceBackend>("matmul", workspace());
    TensorShape inputShape({ 1, 10 }, DataLayout::NC);
    Tensor* input = new Tensor("input", inputShape);
    input->allocateStorage<float>();
    // Input data looks like:
    input->fillData<float>({ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10 });
    workspace()->addTensor(input);
    matMulOp->setInput(input, 0);
    SECTION("10x10, constant weights per neuron") {
        matMulOp->setNumOutputs(10);
        matMulOp->createAllTensors();
        allocateAllTensors<float>(matMulOp);
        auto weightsTensor = matMulOp->getInput(1);
        weightsTensor->fillData<float>({
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        });
        SECTION("Running FC without fused activations.") {
            // Expected output (with zero padding):
            //
            // (1...10) * 1 = 55
            // (1...10) * 2 = 110
            // (1...10) * 3 = 165
            // (1...10) * 4 = 220
            // (1...10) * 5 = 275
            // (1...10) * 6 = 330
            // (1...10) * 7 = 385
            // (1...10) * 8 = 440
            // (1...10) * 9 = 495
            // (1...10) * 10 = 550
            std::vector<float> expectedValues{ -55,  -110, -165, -220, -275,
                                               -330, -385, -440, -495, -550 };
            SECTION("Non-transposed weights") {
                matMulOp->run();
                auto outputsTensor = matMulOp->getOutput(0);
                verifyOutputs(outputsTensor, expectedValues);
            }
            SECTION("Transposed weights") {
                auto transposedWeightsTensor =
                        transposeWeights(weightsTensor, workspace());
                matMulOp->setInput(transposedWeightsTensor, 1);
                matMulOp->run();
                auto outputsTensor = matMulOp->getOutput(0);
                verifyOutputs(outputsTensor, expectedValues);
            }
        }
        SECTION("Running FC with fused activations.") {
            // All negative values are scaled by 0.1.
            std::vector<float> expectedValues{ -5.5, -11, -16.5, -22, -27.5,
                                               -33, -38.5, -44, -49.5, -55 };
            ActivationInfo actInfo;
            actInfo.function = activation_type::LRELU;
            actInfo.params.slope = 0.1;
            matMulOp->setActivation(actInfo);
            SECTION("Non-transposed weights") {
                matMulOp->run();
                auto outputsTensor = matMulOp->getOutput(0);
                verifyOutputs(outputsTensor, expectedValues);
            }
            SECTION("Transposed weights") {
                auto transposedWeightsTensor =
                        transposeWeights(weightsTensor, workspace());
                matMulOp->setInput(transposedWeightsTensor, 1);
                matMulOp->run();
                auto outputsTensor = matMulOp->getOutput(0);
                verifyOutputs(outputsTensor, expectedValues);
            }
        }
    }

    SECTION("10x10, distinct weights per neuron") {
        matMulOp->setNumOutputs(10);
        matMulOp->createAllTensors();
        allocateAllTensors<float>(matMulOp);
        auto weightsTensor = matMulOp->getInput(1);
        weightsTensor->fillData<float>({
            1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
            2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
            3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
            4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
            5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
            6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
            7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
            8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
            9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19
        });
        SECTION("Running FC without fused activations.") {
            // Expected output:
            //
            // 1*1 + 2*2 + 3*3 + ... 10*10 = 385
            // 1*2 + 2*3 + 3*4 + ... 10*11 = 385 + (1+...+10) = 385+55 = 440
            // 1*3 + 2*4 + 3*4 + ... 10*11 = 440 + 55 = 495
            // 1*4 + 2*3 + 3*4 + ... 10*11 = 550
            // 1*5 + 2*3 + 3*4 + ... 10*11 = 605
            // 1*6 + 2*3 + 3*4 + ... 10*11 = 660
            // 1*7 + 2*3 + 3*4 + ... 10*11 = 715
            // 1*8 + 2*3 + 3*4 + ... 10*11 = 770
            // 1*9 + 2*3 + 3*4 + ... 10*11 = 825
            // 1*10 + 2*3 + 3*4 + ... 10*11 = 880
            // 385 440 495 550 605 660 715 770 825 880
            std::vector<float> expectedValues{ -385, -440, -495, -550, -605,
                                               -660, -715, -770, -825, -880 };
            SECTION("Non-transposed weights") {
                matMulOp->run();
                auto outputsTensor = matMulOp->getOutput(0);
                verifyOutputs(outputsTensor, expectedValues);
            }
            SECTION("Transposed weights") {
                auto transposedWeightsTensor =
                        transposeWeights(weightsTensor, workspace());
                matMulOp->setInput(transposedWeightsTensor, 1);
                matMulOp->run();
                auto outputsTensor = matMulOp->getOutput(0);
                verifyOutputs(outputsTensor, expectedValues);
            }
        }
        SECTION("Running FC with fused activations.") {
            // All negative values are scaled by 0.1.
            std::vector<float> expectedValues{ -38.5, -44, -49.5, -55, -60.5,
                                               -66, -71.5, -77, -82.5, -88 };
            ActivationInfo actInfo;
            actInfo.function = activation_type::LRELU;
            actInfo.params.slope = 0.1;
            matMulOp->setActivation(actInfo);
            SECTION("Non-transposed weights") {
                matMulOp->run();
                auto outputsTensor = matMulOp->getOutput(0);
                verifyOutputs(outputsTensor, expectedValues);
            }
            SECTION("Transposed weights") {
                auto transposedWeightsTensor =
                        transposeWeights(weightsTensor, workspace());
                matMulOp->setInput(transposedWeightsTensor, 1);
                matMulOp->run();
                auto outputsTensor = matMulOp->getOutput(0);
                verifyOutputs(outputsTensor, expectedValues);
            }
        }
    }
}
