#include "catch.hpp"
#include "core/backend.h"
#include "core/tensor.h"
#include "core/smaug_test.h"
#include "operators/smv/smv_test_common.h"
#include "operators/smv/smv_convolution_op.h"
#include "operators/smv/smv_convolution_tiling.h"

using namespace smaug;

TEST_CASE_METHOD(SmaugTest, "Basic tiling tests", "[smvtiling]") {
    using namespace smaug::smv;
    using namespace smaug::smv::conv;
    auto convOp = new SmvConvolutionOp("conv", workspace());
    // Outputs should be the same size as inputs.
    convOp->setStride(1, 1);
    convOp->setPadding(SamePadding);

    SECTION("No tiling needed") {
        TensorShape inputShape(
                { 1, 32, 32, 8 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        convOp->setInput(inputs, 0);
        convOp->setWeightDims(5, 5, 8);
        convOp->createAllTensors();
        allocateAllTensors<float16>(convOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(convOp);
        REQUIRE(config.inputs == inputShape);
        REQUIRE(config.weights.dims() == std::vector<int>{ 8, 5, 5, 8 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 32, 32, 8 });
    }

    SECTION("DimNH tiling on inputs when less than 32 channels") {
        TensorShape inputShape(
                { 1, 32, 64, 16 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        convOp->setInput(inputs, 0);
        convOp->setWeightDims(3, 3, 8);
        convOp->createAllTensors();
        allocateAllTensors<float16>(convOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(convOp);
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 16, 64, 16 });
        REQUIRE(config.weights.dims() == std::vector<int>{ 8, 3, 3, 16 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 16, 64, 8 });

        SECTION("Generated tiles have correct shape and data") {
            fillTensorWithFixedData(inputs);
            TiledTensor inputTiles = generateTiledTensor(
                    inputs, config.inputs, { 0, 2, 0, 0 }, workspace());
            REQUIRE(inputTiles.size() == 3);
            for (auto i = inputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = inputTiles[i]->getShape().dims();
                if (i < inputTiles.size() - 1)
                    REQUIRE(testDims == std::vector<int>{ 1, 16, 64, 16 });
                else
                    REQUIRE(testDims == std::vector<int>{ 1, 4, 64, 16 });
                verifyTensorWithFixedData(inputTiles[i], 0);
            }

            auto weights = convOp->getInput(1);
            TiledTensor weightTiles = generateTiledTensor(
                    weights, config.weights, { 0, 0, 0, 0 }, workspace());
            auto outputs = convOp->getOutput(0);
            fillTensorWithFixedData(outputs);
            TiledTensor outputTiles =
                    TilingOptimizer::generateRowwiseOutputTiledTensor(
                            convOp, inputTiles, weightTiles, config.outputs,
                            outputs, true);

            REQUIRE(outputTiles.size() == 3);
            REQUIRE(outputTiles[0]->getShape().dims() ==
                    std::vector<int>{ 1, 15, 64, 8 });
            REQUIRE(outputTiles[1]->getShape().dims() ==
                    std::vector<int>{ 1, 14, 64, 8 });
            REQUIRE(outputTiles[2]->getShape().dims() ==
                    std::vector<int>{ 1, 3, 64, 8 });
            for (auto i = outputTiles.startIndex(); !i.end(); ++i) {
                verifyTensorWithFixedData(outputTiles[i], 0);
            }
        }
    }

    SECTION("DimNC tiling on inputs and weights with > 32 channels") {
        // Pick 16x16 plane size so that there are more possibilities for
        // channel sizes than just one.
        TensorShape inputShape(
                { 1, 16, 16, 128 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        convOp->setInput(inputs, 0);
        convOp->setWeightDims(3, 3, 16);
        convOp->createAllTensors();
        allocateAllTensors<float16>(convOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(convOp);
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 16, 16, 64 });
        REQUIRE(config.weights.dims() == std::vector<int>{ 16, 3, 3, 64 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 16, 16, 16 });

        SECTION("Generated tiles have correct shape and data") {
            fillTensorWithFixedData(inputs);
            TiledTensor inputTiles = generateTiledTensor(
                    inputs, config.inputs, { 0, 2, 0, 0 }, workspace());
            REQUIRE(inputTiles.size() == 2);
            for (auto i = inputTiles.startIndex(); !i.end(); ++i) {
                REQUIRE(inputTiles[i]->getShape().dims() ==
                        std::vector<int>{ 1, 16, 16, 64 });
                verifyTensorWithFixedData(inputTiles[i], i * 64);
            }
        }
    }

    SECTION("DimNCH tiling on inputs") {
        SECTION("32x32 inputs: 3 tiles rowwise, 6 tiles channelwise") {
            TensorShape inputShape({ 1, 32, 32, 192 }, DataLayout::NHWC,
                                   SmvBackend::Alignment);
            Tensor* inputs = new Tensor("inputs", inputShape);
            workspace()->addTensor(inputs);
            convOp->setInput(inputs, 0);
            convOp->setWeightDims(4, 4, 32);
            convOp->createAllTensors();
            allocateAllTensors<float16>(convOp);
            TilingConfig config =
                    TilingOptimizer::computeBasicTileShapes(convOp);
            REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 16, 32, 32 });
            REQUIRE(config.weights.dims() == std::vector<int>{ 32, 4, 4, 32 });
            REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 16, 32, 32 });

            SECTION("Generated tiles have correct shape and data") {
                fillTensorWithFixedData(inputs);
                TiledTensor inputTiles = generateTiledTensor(
                        inputs, config.inputs, { 0, 3, 0, 0 }, workspace());
                REQUIRE(inputTiles.size() == 18);
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 6; j++) {
                        int index = i * 6 + j;
                        if (i < 2) {
                            REQUIRE(inputTiles[index]->getShape().dims() ==
                                    std::vector<int>{ 1, 16, 32, 32 });
                        } else {
                            REQUIRE(inputTiles[index]->getShape().dims() ==
                                    std::vector<int>{ 1, 6, 32, 32 });
                        }
                        verifyTensorWithFixedData(inputTiles[index], j * 32);
                    }
                }

                auto weights = convOp->getInput(1);
                fillTensorWithFixedData(weights);
                TiledTensor weightTiles = generateTiledTensor(
                        weights, config.weights, { 0, 0, 0, 0 }, workspace());
                REQUIRE(weightTiles.size() == 6);
                for (auto i = weightTiles.startIndex(); !i.end(); ++i) {
                    REQUIRE(weightTiles[i]->getShape().dims() ==
                            std::vector<int>{ 32, 4, 4, 32 });
                    verifyTensorWithFixedData(weightTiles[i], i * 32);
                }

                auto outputs = convOp->getOutput(0);
                fillTensorWithFixedData(outputs);
                TiledTensor outputTiles =
                        TilingOptimizer::generateRowwiseOutputTiledTensor(
                                convOp, inputTiles, weightTiles, config.outputs,
                                outputs, true);
                REQUIRE(outputTiles.size() == 3);
                for (auto i = outputTiles.startIndex(); !i.end(); ++i) {
                    if (i == 0) {
                        REQUIRE(outputTiles[i]->getShape().dims() ==
                                std::vector<int>{ 1, 15, 32, 32 });
                    } else if (i == 1) {
                        REQUIRE(outputTiles[i]->getShape().dims() ==
                                std::vector<int>{ 1, 13, 32, 32 });
                    } else {
                        REQUIRE(outputTiles[i]->getShape().dims() ==
                                std::vector<int>{ 1, 4, 32, 32 });
                    }
                    verifyTensorWithFixedData(outputTiles[i], 0);
                }
            }
        }
        SECTION("64x64 inputs: 9 tiles rowwise, 6 tiles channelwise") {
            TensorShape inputShape({ 1, 64, 64, 192 }, DataLayout::NHWC,
                                   SmvBackend::Alignment);
            Tensor* inputs = new Tensor("inputs", inputShape);
            workspace()->addTensor(inputs);
            convOp->setInput(inputs, 0);
            convOp->setWeightDims(2, 2, 32);
            convOp->createAllTensors();
            allocateAllTensors<float16>(convOp);
            TilingConfig config =
                    TilingOptimizer::computeBasicTileShapes(convOp);
            REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 8, 64, 32 });
            REQUIRE(config.weights.dims() == std::vector<int>{ 32, 2, 2, 32 });
            REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 8, 64, 32 });

            SECTION("Generated tiles have correct shape and data") {
                fillTensorWithFixedData(inputs);
                TiledTensor inputTiles = generateTiledTensor(
                        inputs, config.inputs, { 0, 1, 0, 0 }, workspace());
                REQUIRE(inputTiles.size() == 54);
                for (int i = 0; i < 9; i++) {
                    for (int j = 0; j < 6; j++) {
                        int index = i * 6 + j;
                        REQUIRE(inputTiles[index]->getShape().dims() ==
                                std::vector<int>{ 1, 8, 64, 32 });
                        verifyTensorWithFixedData(inputTiles[index], j * 32);
                    }
                }

                auto weights = convOp->getInput(1);
                fillTensorWithFixedData(weights);
                TiledTensor weightTiles = generateTiledTensor(
                        weights, config.weights, { 0, 0, 0, 0 }, workspace());
                REQUIRE(weightTiles.size() == 6);
                for (auto i = weightTiles.startIndex(); !i.end(); ++i) {
                    REQUIRE(weightTiles[i]->getShape().dims() ==
                            std::vector<int>{ 32, 2, 2, 32 });
                    verifyTensorWithFixedData(weightTiles[i], i * 32);
                }

                auto outputs = convOp->getOutput(0);
                fillTensorWithFixedData(outputs);
                TiledTensor outputTiles =
                        TilingOptimizer::generateRowwiseOutputTiledTensor(
                                convOp, inputTiles, weightTiles, config.outputs,
                                outputs, true);
                REQUIRE(outputTiles.size() == 9);
                for (auto i = outputTiles.startIndex(); !i.end(); ++i) {
                    if (i == 0) {
                        REQUIRE(outputTiles[i]->getShape().dims() ==
                                std::vector<int>{ 1, 8, 64, 32 });
                    } else {
                        REQUIRE(outputTiles[i]->getShape().dims() ==
                                std::vector<int>{ 1, 7, 64, 32 });
                    }
                    verifyTensorWithFixedData(outputTiles[i], 0);
                }
            }
        }
        SECTION("128x128 inputs: 43 tiles rowwise, 6 tiles channelwise") {
            TensorShape inputShape({ 1, 128, 128, 192 }, DataLayout::NHWC,
                                   SmvBackend::Alignment);
            Tensor* inputs = new Tensor("inputs", inputShape);
            workspace()->addTensor(inputs);
            convOp->setInput(inputs, 0);
            convOp->setWeightDims(2, 2, 32);
            convOp->createAllTensors();
            allocateAllTensors<float16>(convOp);
            TilingConfig config =
                    TilingOptimizer::computeBasicTileShapes(convOp);
            REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 4, 128, 32 });
            REQUIRE(config.weights.dims() == std::vector<int>{ 32, 2, 2, 32 });
            REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 4, 128, 32 });

            SECTION("Generated tiles have correct shape and data") {
                fillTensorWithFixedData(inputs);
                TiledTensor inputTiles = generateTiledTensor(
                        inputs, config.inputs, { 0, 1, 0, 0 }, workspace());
                REQUIRE(inputTiles.size() == 258);
                for (int i = 0; i < 43; i++) {
                    for (int j = 0; j < 6; j++) {
                        int index = i * 6 + j;
                        if (i < 42) {
                            REQUIRE(inputTiles[index]->getShape().dims() ==
                                    std::vector<int>{ 1, 4, 128, 32 });
                        } else {
                            REQUIRE(inputTiles[index]->getShape().dims() ==
                                    std::vector<int>{ 1, 2, 128, 32 });
                        }
                        verifyTensorWithFixedData(inputTiles[index], j * 32);
                    }
                }

                auto weights = convOp->getInput(1);
                fillTensorWithFixedData(weights);
                TiledTensor weightTiles = generateTiledTensor(
                        weights, config.weights, { 0, 0, 0, 0 }, workspace());
                REQUIRE(weightTiles.size() == 6);
                for (auto i = weightTiles.startIndex(); !i.end(); ++i) {
                    REQUIRE(weightTiles[i]->getShape().dims() ==
                            std::vector<int>{ 32, 2, 2, 32 });
                    verifyTensorWithFixedData(weightTiles[i], i * 32);
                }

                auto outputs = convOp->getOutput(0);
                fillTensorWithFixedData(outputs);
                TiledTensor outputTiles =
                        TilingOptimizer::generateRowwiseOutputTiledTensor(
                                convOp, inputTiles, weightTiles, config.outputs,
                                outputs, true);
                REQUIRE(outputTiles.size() == 43);
                for (auto i = outputTiles.startIndex(); !i.end(); ++i) {
                    if (i == 0) {
                        REQUIRE(outputTiles[i]->getShape().dims() ==
                                std::vector<int>{ 1, 4, 128, 32 });
                    } else if (i < 42) {
                        REQUIRE(outputTiles[i]->getShape().dims() ==
                                std::vector<int>{ 1, 3, 128, 32 });
                    } else {
                        REQUIRE(outputTiles[i]->getShape().dims() ==
                                std::vector<int>{ 1, 1, 128, 32 });
                    }
                    verifyTensorWithFixedData(outputTiles[i], 0);
                }
            }
        }
    }

    SECTION("DimN tiling for weights, None for inputs") {
        // Inputs can all fit.
        TensorShape inputShape(
                { 1, 8, 8, 192 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        convOp->setInput(inputs, 0);
        // ...but weights can't. The scratchpad can fit up to 9 complete
        // filters at once, so we just need DimN tiling.
        convOp->setWeightDims(3, 3, 256);
        convOp->createAllTensors();
        allocateAllTensors<float16>(convOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(convOp);
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 8, 8, 192 });
        REQUIRE(config.weights.dims() == std::vector<int>{ 8, 3, 3, 192 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 8, 8, 8 });

        SECTION("Generated tiles have correct shape and data") {
            TiledTensor inputTiles = generateTiledTensor(
                    inputs, config.inputs, { 0, 2, 0, 0 }, workspace());
            REQUIRE(inputTiles.size() == 1);

            auto weights = convOp->getInput(1);
            fillTensorWithFixedData(weights);
            TiledTensor weightTiles = generateTiledTensor(
                    weights, config.weights, { 0, 0, 0, 0 }, workspace());
            REQUIRE(weightTiles.size() == 256 / 8);
            for (auto i = weightTiles.startIndex(); !i.end(); ++i) {
                REQUIRE(weightTiles[i]->getShape().dims() ==
                        std::vector<int>{ 8, 3, 3, 192 });
                verifyTensorWithFixedData(weightTiles[i], 0);
            }

            auto outputs = convOp->getOutput(0);
            fillTensorWithFixedData(outputs);
            TiledTensor outputTiles = generateTiledTensor(
                    outputs, config.outputs, { 0, 2, 0, 0 }, workspace());
            REQUIRE(outputTiles.size() == 256 / 8);
            for (auto i = outputTiles.startIndex(); !i.end(); ++i) {
                REQUIRE(outputTiles[i]->getShape().dims() ==
                        std::vector<int>{ 1, 8, 8, 8 });
                verifyTensorWithFixedData(outputTiles[i], i * 8);
            }
        }
    }

    SECTION("DimNC tiling for weights, None for inputs") {
        // Inputs can all fit.
        TensorShape inputShape(
                { 1, 8, 8, 256 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        convOp->setInput(inputs, 0);
        // ...but weights can't. The scratchpad can only fit up to 7 complete
        // filters at once, which is smaller than the minimum 8 required for
        // DimN tiling, so we'll need to go to DimNC.
        convOp->setWeightDims(3, 3, 256);
        convOp->createAllTensors();
        allocateAllTensors<float16>(convOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(convOp);
        // Inputs don't need to be tiled.
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 8, 8, 256 });
        // In terms of weight utilization, 56x3x3x32 is the same as 8x3x3x224,
        // but it allows us to store up to 56 partial output feature maps
        // instead of just 8, so the overall utilization of the SRAMs is
        // higher.
        REQUIRE(config.weights.dims() == std::vector<int>{ 56, 3, 3, 32 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 8, 8, 56 });

        SECTION("Generated tiles have correct shape and data") {
            TiledTensor inputTiles = generateTiledTensor(
                    inputs, config.inputs, { 0, 2, 0, 0 }, workspace());
            REQUIRE(inputTiles.size() == 1);

            auto weights = convOp->getInput(1);
            fillTensorWithFixedData(weights);
            TiledTensor weightTiles = generateTiledTensor(
                    weights, config.weights, { 0, 0, 0, 0 }, workspace());
            // 256 output channels in groups of 56 means 4 groups of 56 and 1
            // group of 32 -> total of 5.
            // 256 input channels in groups of 32 -> 8 groups of 32.
            REQUIRE(weightTiles.size() == 5 * 8);
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 8; j++) {
                    int index = i * 8 + j;
                    const std::vector<int>& tileDims =
                            weightTiles[index]->getShape().dims();
                    if (i < 4) {
                        REQUIRE(tileDims == std::vector<int>{ 56, 3, 3, 32 });
                    } else {
                        REQUIRE(tileDims == std::vector<int>{ 32, 3, 3, 32 });
                    }
                    verifyTensorWithFixedData(weightTiles[index], j * 32);
                }
            }

            auto outputs = convOp->getOutput(0);
            fillTensorWithFixedData(outputs);
            TiledTensor outputTiles = generateTiledTensor(
                    outputs, config.outputs, { 0, 2, 0, 0 }, workspace());
            REQUIRE(outputTiles.size() == ceil(256.0 / 56));
            for (auto i = outputTiles.startIndex(); !i.end(); ++i) {
                const std::vector<int>& tileDims =
                        outputTiles[i]->getShape().dims();
                if (i < outputTiles.size() - 1) {
                    REQUIRE(tileDims == std::vector<int>{ 1, 8, 8, 56 });
                } else {
                    REQUIRE(tileDims == std::vector<int>{ 1, 8, 8, 32 });
                }
                verifyTensorWithFixedData(outputTiles[i], i * 56);
            }

        }
    }
}

TEST_CASE_METHOD(SmaugTest, "Kernel shape tests", "[smvtiling]") {
    using namespace smaug::smv;
    using namespace smaug::smv::conv;
    auto convOp = new SmvConvolutionOp("conv", workspace());
    // Outputs should be the same size as inputs.
    convOp->setStride(1, 1);
    convOp->setPadding(SamePadding);

    SECTION("1x1 kernels") {
        TensorShape inputShape(
                { 1, 32, 32, 8 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        convOp->setInput(inputs, 0);
        convOp->setWeightDims(1, 1, 128);
        convOp->createAllTensors();
        allocateAllTensors<float16>(convOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(convOp);
        REQUIRE(config.inputs == inputShape);
        REQUIRE(config.weights.dims() == std::vector<int>{ 128, 1, 1, 8 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 32, 32, 16 });

        SECTION("Weights are not tiled but outputs are.") {
            auto weights = convOp->getInput(1);
            fillTensorWithFixedData(weights);
            TiledTensor weightTiles = generateTiledTensor(
                    weights, config.weights, { 0, 0, 0, 0 }, workspace());
            REQUIRE(weightTiles.size() == 1);
            // Don't need to verify shape again.
            verifyTensorWithFixedData(weightTiles[0], 0);

            auto outputs = convOp->getOutput(0);
            fillTensorWithFixedData(outputs);
            TiledTensor outputTiles = generateTiledTensor(
                    outputs, config.outputs, { 0, 0, 0, 0 }, workspace());
            REQUIRE(outputTiles.size() == 128 / 16);
            for (auto i = outputTiles.startIndex(); !i.end(); ++i) {
                REQUIRE(outputTiles[i]->getShape().dims() ==
                        std::vector<int>{ 1, 32, 32, 16 });
                verifyTensorWithFixedData(outputTiles[i], i * 16);
            }
        }
    }

    SECTION("2x2 kernels") {
        TensorShape inputShape(
                { 1, 32, 64, 32 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        convOp->setInput(inputs, 0);
        convOp->setWeightDims(2, 2, 1024);
        convOp->createAllTensors();
        allocateAllTensors<float16>(convOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(convOp);
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 2, 64, 32 });
        REQUIRE(config.weights.dims() == std::vector<int>{ 128, 2, 2, 32 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 2, 64, 128 });

        SECTION("Inputs and outputs tiled DimNH, weights tiled DimN") {
            fillTensorWithFixedData(inputs);
            TiledTensor inputTiles = generateTiledTensor(
                    inputs, config.inputs, { 0, 1, 0, 0 }, workspace());
            // Halo size 1: 0-1, 1-2, 2-3, ... 30-31, total 31 tiles.
            REQUIRE(inputTiles.size() == 31);
            for (auto i = inputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = inputTiles[i]->getShape().dims();
                REQUIRE(testDims == std::vector<int>{ 1, 2, 64, 32 });
                verifyTensorWithFixedData(inputTiles[i], 0);
            }

            auto weights = convOp->getInput(1);
            fillTensorWithFixedData(weights);
            TiledTensor weightTiles = generateTiledTensor(
                    weights, config.weights, { 0, 0, 0, 0 }, workspace());
            REQUIRE(weightTiles.size() == 8);
            for (int i = 0; i < weightTiles.size(); ++i)
                verifyTensorWithFixedData(weightTiles[0], 0);

            auto outputs = convOp->getOutput(0);
            fillTensorWithFixedData(outputs);
            TiledTensor outputTiles =
                    TilingOptimizer::generateRowwiseOutputTiledTensor(
                            convOp, inputTiles, weightTiles, config.outputs,
                            outputs, true);
            // There are 31 tiles in the rowwise dimension and 8 in the output
            // channel dim.
            REQUIRE(outputTiles.size() == 8*31);
            for (int r = 0; r < 31; r++) {
                for (int c = 0; c < 8; c++) {
                    int idx = r * 8 + c;
                    auto& testDims = outputTiles[idx]->getShape().dims();
                    // Since we use same padding here, for this 2x2 kernel size,
                    // top padding is 1 and bottom padding is 0.
                    if (r == 0) {
                        // Top tile. Top padding size 1. Input tile row size 2.
                        // Output tile row size is 2 because of the zero-padded
                        // row at the top.
                        REQUIRE(testDims == std::vector<int>{ 1, 2, 64, 128 });
                    } else if (r < 31) {
                        // Middle/bottom tiles. No top/bottom padding. Input
                        // tile row size 2. Output tile row size 1.
                        REQUIRE(testDims == std::vector<int>{ 1, 1, 64, 128 });
                    }
                    verifyTensorWithFixedData(outputTiles[idx], c * 128);
                }
            }
        }
    }

    SECTION("5x5 kernels") {
        TensorShape inputShape(
                { 1, 64, 32, 32 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        convOp->setInput(inputs, 0);
        convOp->setWeightDims(5, 5, 32);
        convOp->createAllTensors();
        allocateAllTensors<float16>(convOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(convOp);
        // The inputs must have at least five rows (same as weights).
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 16, 32, 32 });
        REQUIRE(config.weights.dims() == std::vector<int>{ 16, 5, 5, 32 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 16, 32, 16 });

        SECTION("Test if the halo region of 2 is handled") {
            fillTensorWithFixedData(inputs);
            TiledTensor inputTiles = generateTiledTensor(
                    inputs, config.inputs, { 0, 2, 0, 0 }, workspace());
            // Halo size 2: 0-15, 14-29, 28-43, 42-57, 56-63
            REQUIRE(inputTiles.size() == 5);
            for (auto i = inputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = inputTiles[i]->getShape().dims();
                if (i < 4)
                    REQUIRE(testDims == std::vector<int>{ 1, 16, 32, 32 });
                else
                    REQUIRE(testDims == std::vector<int>{ 1, 8, 32, 32 });
                verifyTensorWithFixedData(inputTiles[i], 0);
            }

            auto weights = convOp->getInput(1);
            fillTensorWithFixedData(weights);
            TiledTensor weightTiles = generateTiledTensor(
                    weights, config.weights, { 0, 0, 0, 0 }, workspace());
            REQUIRE(weightTiles.size() == 2);
            for (auto i = weightTiles.startIndex(); !i.end(); ++i)
                verifyTensorWithFixedData(weightTiles[0], 0);

            auto outputs = convOp->getOutput(0);
            fillTensorWithFixedData(outputs);
            TiledTensor outputTiles = generateTiledTensor(
                    outputs, config.outputs, { 0, 2, 0, 0 }, workspace());
            // 5 tiles in rowwise direction, 2 in channelwise.
            REQUIRE(outputTiles.size() == 5 * 2);
            for (int r = 0; r < 5; r++) {
                for (int c = 0; c < 2; c++) {
                    int idx = r * 2 + c;
                    auto& testDims = outputTiles[idx]->getShape().dims();
                    if (r < 4)
                        REQUIRE(testDims == std::vector<int>{ 1, 16, 32, 16 });
                    else
                        REQUIRE(testDims == std::vector<int>{ 1, 8, 32, 16 });
                    verifyTensorWithFixedData(outputTiles[idx], c * 16);
                }
            }
        }
    }
}

TEST_CASE_METHOD(SmaugTest, "Stride size tests", "[smvtiling]") {
    using namespace smaug::smv;
    using namespace smaug::smv::conv;
    auto convOp = new SmvConvolutionOp("conv", workspace());
    convOp->setPadding(ValidPadding);

    SECTION("2x2 strides") {
        // Inputs and outputs use DimNH, None for weights.
        convOp->setStride(2, 2);
        TensorShape inputShape(
                { 1, 64, 64, 32 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        convOp->setInput(inputs, 0);
        convOp->setWeightDims(3, 3, 16);
        convOp->createAllTensors();
        allocateAllTensors<float16>(convOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(convOp);
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 7, 64, 32 });
        REQUIRE(config.weights.dims() == std::vector<int>{ 16, 3, 3, 32 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 3, 31, 16 });

        SECTION("Generated tiles have correct shape and data") {
            fillTensorWithFixedData(inputs);
            TiledTensor inputTiles = generateTiledTensor(
                    inputs, config.inputs, { 0, 1, 0, 0 }, workspace());
            REQUIRE(inputTiles.size() == 11);
            for (auto i = inputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = inputTiles[i]->getShape().dims();
                if (i < 10) {
                    REQUIRE(testDims == config.inputs.dims());
                } else {
                    REQUIRE(testDims == std::vector<int>{ 1, 4, 64, 32 });
                }
                verifyTensorWithFixedData(inputTiles[i], 0);
            }

            auto weights = convOp->getInput(1);
            fillTensorWithFixedData(weights);
            TiledTensor weightTiles = generateTiledTensor(
                    weights, config.weights, { 0, 0, 0, 0 }, workspace());
            REQUIRE(weightTiles.size() == 1);
            verifyTensorWithFixedData(weightTiles[0], 0);

            auto outputs = convOp->getOutput(0);
            fillTensorWithFixedData(outputs);
            TiledTensor outputTiles =
                    TilingOptimizer::generateRowwiseOutputTiledTensor(
                            convOp, inputTiles, weightTiles, config.outputs,
                            outputs, true);
            REQUIRE(outputTiles.size() == 11);
            for (auto i = outputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = outputTiles[i]->getShape().dims();
                if (i < 10) {
                    REQUIRE(testDims == config.outputs.dims());
                } else {
                    REQUIRE(testDims == std::vector<int>{ 1, 1, 31, 16 });
                }
                verifyTensorWithFixedData(outputTiles[i], 0);
            }
        }
    }

    SECTION("3x3 strides") {
        // Inputs and outputs use DimNH, None for weights.
        convOp->setStride(3, 3);
        TensorShape inputShape(
                { 1, 64, 64, 32 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        convOp->setInput(inputs, 0);
        convOp->setWeightDims(5, 5, 16);
        convOp->createAllTensors();
        allocateAllTensors<float16>(convOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(convOp);
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 8, 64, 32 });
        REQUIRE(config.weights.dims() == std::vector<int>{ 16, 5, 5, 32 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 2, 20, 16 });

        SECTION("Generated tiles have correct shape and data") {
            fillTensorWithFixedData(inputs);
            TiledTensor inputTiles = generateTiledTensor(
                    inputs, config.inputs, { 0, 2, 0, 0 }, workspace());
            REQUIRE(inputTiles.size() == 11);
            for (auto i = inputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = inputTiles[i]->getShape().dims();
                if (i < 10) {
                    REQUIRE(testDims == config.inputs.dims());
                } else {
                    REQUIRE(testDims == std::vector<int>{ 1, 4, 64, 32 });
                }
                verifyTensorWithFixedData(inputTiles[i], 0);
            }

            auto weights = convOp->getInput(1);
            fillTensorWithFixedData(weights);
            TiledTensor weightTiles = generateTiledTensor(
                    weights, config.weights, { 0, 0, 0, 0 }, workspace());
            REQUIRE(weightTiles.size() == 1);
            verifyTensorWithFixedData(weightTiles[0], 0);

            auto outputs = convOp->getOutput(0);
            fillTensorWithFixedData(outputs);
            TiledTensor outputTiles =
                    TilingOptimizer::generateRowwiseOutputTiledTensor(
                            convOp, inputTiles, weightTiles, config.outputs,
                            outputs, true);
            REQUIRE(outputTiles.size() == 11);
            for (auto i = outputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = outputTiles[i]->getShape().dims();
                if (i < 10) {
                    REQUIRE(testDims == config.outputs.dims());
                } else {
                    REQUIRE(testDims == std::vector<int>{ 1, 1, 20, 16 });
                }
                verifyTensorWithFixedData(outputTiles[i], 0);
            }
        }
    }
}
