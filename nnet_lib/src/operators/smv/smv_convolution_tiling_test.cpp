#include "catch.hpp"
#include "core/backend.h"
#include "core/tensor.h"
#include "core/smaug_test.h"
#include "operators/smv/smv_convolution_op.h"
#include "operators/smv/smv_convolution_tiling.h"

using namespace smaug;
using namespace smaug::smv::conv;

void fillTensorWithData(SmvTensor* tensor) {
    const TensorShape& shape = tensor->getShape();
    // Each dimension C is iniialized to a different constant value.
    float* dataPtr = tensor->data<float>();
    int resetCounter = shape.getStorageDim(3);
    int value = 0;
    for (int i = 0; i < shape.storageSize(); i++) {
        dataPtr[i] = value++;
        if ((i + 1) % resetCounter == 0)
            value = 0;
    }
}

void verifyTensorData(SmvTensor* tensor, int valueOffset) {
    float* dataPtr = tensor->data<float>();
    int expectedValue = valueOffset;
    int resetCounter = tensor->getShape().getStorageDim(3);
    int totalSize = tensor->getShape().storageSize();
    for (int i = 0; i < totalSize; i++) {
        REQUIRE(dataPtr[i] == expectedValue);
        ++expectedValue;
        if ((i + 1) % resetCounter == 0)
            expectedValue = valueOffset;
    }
}

TEST_CASE_METHOD(SmaugTest, "Basic tiling tests", "[smvtiling]") {
    smv::kSpadSize = 32 * 1024;
    auto convOp = new SmvConvolutionOp("conv", workspace());
    // Outputs should be the same size as inputs.
    convOp->setStride(1, 1);
    convOp->setPadding(SamePadding);

    SECTION("No tiling needed") {
        TensorShape inputShape({ 1, 32, 32, 8 }, DataLayout::NHWC);
        Tensor<SmvBackend>* inputs =
                new Tensor<SmvBackend>("inputs", inputShape);
        convOp->setInput(inputs, 0);
        convOp->setWeightDims(5, 5, 8);
        convOp->createAllTensors();
        allocateAllTensors<float, SmvBackend>(convOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(convOp);
        REQUIRE(config.inputs == inputShape);
        REQUIRE(config.weights.dims() == std::vector<int>{ 8, 5, 5, 8 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 32, 32, 8 });
    }

    SECTION("DimNH tiling on inputs when less than 32 channels") {
        TensorShape inputShape({ 1, 32, 32, 16 }, DataLayout::NHWC);
        Tensor<SmvBackend>* inputs =
                new Tensor<SmvBackend>("inputs", inputShape);
        convOp->setInput(inputs, 0);
        convOp->setWeightDims(3, 3, 8);
        convOp->createAllTensors();
        allocateAllTensors<float, SmvBackend>(convOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(convOp);
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 16, 32, 16 });
        REQUIRE(config.weights.dims() == std::vector<int>{ 8, 3, 3, 16 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 16, 32, 8 });

        SECTION("Generated tiles have correct shape and data") {
            fillTensorWithData(inputs);
            std::vector<SmvTensor*> inputTiles =
                    TilingOptimizer::generateBlockedTensor(
                            inputs, config.inputs, { 0, 1, 1, 0});
            REQUIRE(inputTiles.size() == 3);
            for (int i = 0; i < inputTiles.size(); i++) {
                auto& testDims = inputTiles[i]->getShape().dims();
                if (i < inputTiles.size() - 1)
                    REQUIRE(testDims == std::vector<int>{ 1, 16, 32, 16 });
                else
                    REQUIRE(testDims == std::vector<int>{ 1, 2, 32, 16 });
                verifyTensorData(inputTiles[i], 0);
            }

            auto outputs = convOp->getOutput<SmvBackend>(0);
            fillTensorWithData(outputs);
            std::vector<SmvTensor*> outputTiles =
                    TilingOptimizer::generateBlockedTensor(
                            outputs, config.outputs, { 0, 1, 1, 0});
            REQUIRE(outputTiles.size() == 3);
            for (int i = 0; i < inputTiles.size(); i++) {
                auto& testDims = outputTiles[i]->getShape().dims();
                if (i < outputTiles.size() - 1)
                    REQUIRE(testDims == std::vector<int>{ 1, 16, 32, 8 });
                else
                    REQUIRE(testDims == std::vector<int>{ 1, 2, 32, 8 });
                verifyTensorData(outputTiles[i], 0);
            }
        }
    }

    SECTION("DimNC tiling on inputs and weights with > 32 channels") {
        // Pick 16x8 plane size so that there are more possibilities for channel
        // sizes than just one.
        TensorShape inputShape({ 1, 16, 8, 128 }, DataLayout::NHWC);
        Tensor<SmvBackend>* inputs =
                new Tensor<SmvBackend>("inputs", inputShape);
        convOp->setInput(inputs, 0);
        convOp->setWeightDims(3, 3, 8);
        convOp->createAllTensors();
        allocateAllTensors<float, SmvBackend>(convOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(convOp);
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 16, 8, 64 });
        REQUIRE(config.weights.dims() == std::vector<int>{ 8, 3, 3, 64 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 16, 8, 8 });

        SECTION("Generated tiles have correct shape and data") {
            fillTensorWithData(inputs);
            std::vector<SmvTensor*> inputTiles =
                    TilingOptimizer::generateBlockedTensor(
                            inputs, config.inputs, { 0, 1, 1, 0});
            REQUIRE(inputTiles.size() == 2);
            for (int i = 0; i < inputTiles.size(); i++) {
                REQUIRE(inputTiles[i]->getShape().dims() ==
                        std::vector<int>{ 1, 16, 8, 64});
                verifyTensorData(inputTiles[i], i * 64);
            }
        }
    }

    SECTION("DimN tiling for weights, None for inputs") {
        // Inputs can all fit.
        TensorShape inputShape({ 1, 8, 8, 96}, DataLayout::NHWC);
        Tensor<SmvBackend>* inputs =
                new Tensor<SmvBackend>("inputs", inputShape);
        convOp->setInput(inputs, 0);
        // ...but weights can't. The scratchpad can fit up to 9 complete
        // filters at once, so we just need DimN tiling.
        convOp->setWeightDims(3, 3, 128);
        convOp->createAllTensors();
        allocateAllTensors<float, SmvBackend>(convOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(convOp);
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 8, 8, 96 });
        REQUIRE(config.weights.dims() == std::vector<int>{ 8, 3, 3, 96 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 8, 8, 8 });

        SECTION("Generated tiles have correct shape and data") {
            std::vector<SmvTensor*> inputTiles =
                    TilingOptimizer::generateBlockedTensor(
                            inputs, config.inputs, { 0, 1, 1, 0});
            REQUIRE(inputTiles.size() == 1);

            auto weights = convOp->getInput<SmvBackend>(1);
            fillTensorWithData(weights);
            std::vector<SmvTensor*> weightTiles =
                    TilingOptimizer::generateBlockedTensor(
                            weights, config.weights, { 0, 0, 0, 0 });
            REQUIRE(weightTiles.size() == 128 / 8);
            for (int i = 0; i < weightTiles.size(); i++) {
                REQUIRE(weightTiles[i]->getShape().dims() ==
                        std::vector<int>{ 8, 3, 3, 96 });
                verifyTensorData(weightTiles[i], 0);
            }

            auto outputs = convOp->getOutput<SmvBackend>(0);
            fillTensorWithData(outputs);
            std::vector<SmvTensor*> outputTiles =
                    TilingOptimizer::generateBlockedTensor(
                            outputs, config.outputs, { 0, 1, 1, 0});
            REQUIRE(outputTiles.size() == 128 / 8);
            for (int i = 0; i < outputTiles.size(); i++) {
                REQUIRE(outputTiles[i]->getShape().dims() ==
                        std::vector<int>{ 1, 8, 8, 8 });
                verifyTensorData(outputTiles[i], i * 8);
            }
        }
    }

    SECTION("DimNC tiling for weights, None for inputs") {
        // Inputs can all fit.
        TensorShape inputShape({ 1, 8, 8, 128}, DataLayout::NHWC);
        Tensor<SmvBackend>* inputs =
                new Tensor<SmvBackend>("inputs", inputShape);
        convOp->setInput(inputs, 0);
        // ...but weights can't. The scratchpad can only fit up to 7 complete
        // filters at once, which is smaller than the minimum 8 required for
        // DimN tiling, so we'll need to go to DimNC.
        convOp->setWeightDims(3, 3, 128);
        convOp->createAllTensors();
        allocateAllTensors<float, SmvBackend>(convOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(convOp);
        // Inputs don't need to be tiled.
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 8, 8, 128 });
        // In terms of weight utilization, 24x3x3x32 is the same as 8x3x3x96,
        // but it allows us to store up to 24 partial output feature maps
        // instead of just 8, so the overall utilization of the SRAMs is
        // higher.
        REQUIRE(config.weights.dims() == std::vector<int>{ 24, 3, 3, 32 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 8, 8, 24 });

        SECTION("Generated tiles have correct shape and data") {
            std::vector<SmvTensor*> inputTiles =
                    TilingOptimizer::generateBlockedTensor(
                            inputs, config.inputs, { 0, 1, 1, 0});
            REQUIRE(inputTiles.size() == 1);

            auto weights = convOp->getInput<SmvBackend>(1);
            fillTensorWithData(weights);
            std::vector<SmvTensor*> weightTiles =
                    TilingOptimizer::generateBlockedTensor(
                            weights, config.weights, { 0, 0, 0, 0 });
            // 128 output channels in groups of 24 means 5 groups of 24 and 1
            // group of 8 -> total of 6.
            // 128 input channels in groups of 32 -> 4 groups of 32.
            REQUIRE(weightTiles.size() == 4*6);
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 4; j++) {
                    int index = i * 4 + j;
                    const std::vector<int>& tileDims =
                            weightTiles[index]->getShape().dims();
                    if (i < 5) {
                        REQUIRE(tileDims == std::vector<int>{ 24, 3, 3, 32 });
                    } else {
                        REQUIRE(tileDims == std::vector<int>{ 8, 3, 3, 32 });
                    }
                    verifyTensorData(weightTiles[index], j * 32);
                }
            }

            auto outputs = convOp->getOutput<SmvBackend>(0);
            fillTensorWithData(outputs);
            std::vector<SmvTensor*> outputTiles =
                    TilingOptimizer::generateBlockedTensor(
                            outputs, config.outputs, { 0, 1, 1, 0});
            REQUIRE(outputTiles.size() == ceil(128.0 / 24));
            for (int i = 0; i < outputTiles.size(); i++) {
                const std::vector<int>& tileDims =
                        outputTiles[i]->getShape().dims();
                if (i < outputTiles.size() - 1) {
                    REQUIRE(tileDims == std::vector<int>{ 1, 8, 8, 24 });
                } else {
                    REQUIRE(tileDims == std::vector<int>{ 1, 8, 8, 8 });
                }
                verifyTensorData(outputTiles[i], i * 24);
            }

        }
    }
}
