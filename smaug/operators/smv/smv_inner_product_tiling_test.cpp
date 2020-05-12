#include "catch.hpp"
#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/core/smaug_test.h"
#include "smaug/operators/smv/smv_test_common.h"
#include "smaug/operators/smv/smv_inner_product_op.h"
#include "smaug/operators/smv/smv_inner_product_tiling.h"

using namespace smaug;

TEST_CASE_METHOD(SmaugTest, "Basic tiling tests", "[smvtiling]") {
    using namespace smaug::smv;
    using namespace smaug::smv::fc;
    auto fcOp = new SmvInnerProductOp("fc", workspace());

    SECTION("No tiling needed") {
        TensorShape inputShape(
                { 1, 256 }, DataLayout::NC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        fcOp->setInput(inputs, 0);
        fcOp->setNumOutputs(32);
        fcOp->createAllTensors();
        allocateAllTensors<float16>(fcOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(fcOp);
        REQUIRE(config.inputs == inputShape);
        REQUIRE(config.weights.dims() == std::vector<int>{ 32, 256 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 32 });
    }

    SECTION("DimN tiling for weights, None for inputs") {
        // Inputs can all fit.
        TensorShape inputShape(
                { 1, 256 }, DataLayout::NC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        fcOp->setInput(inputs, 0);
        // ...but weights can't. Weights will be tiled into 2 neuron-wise tiles.
        fcOp->setNumOutputs(128);
        fcOp->createAllTensors();
        allocateAllTensors<float16>(fcOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(fcOp);
        REQUIRE(config.inputs == inputShape);
        REQUIRE(config.weights.dims() == std::vector<int>{ 64, 256 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 128 });

        SECTION("Generated tiles have correct shape and data") {
            fillTensorWithFixedData(inputs);
            TiledTensor inputTiles =
                    generateTiledTensorAndCopyData(inputs, config.inputs, fcOp);
            REQUIRE(inputTiles.size() == 1);
            verifyTensorWithFixedData(inputTiles[0], 0);

            auto weights = fcOp->getInput(1);
            fillTensorWithFixedData(weights);
            TiledTensor weightTiles = generateTiledTensorAndCopyData(
                    weights, config.weights, fcOp);
            REQUIRE(weightTiles.size() == 2);
            for (auto i = weightTiles.startIndex(); !i.end(); ++i) {
                REQUIRE(weightTiles[i]->getShape().dims() ==
                        std::vector<int>{ 64, 256 });
                verifyTensorWithFixedData(weightTiles[i], 0);
            }

            auto outputs = fcOp->getOutput(0);
            fillTensorWithFixedData(outputs);
            TiledTensor outputTiles = generateTiledTensorAndCopyData(
                    outputs, config.outputs, fcOp);
            REQUIRE(outputTiles.size() == 1);
            verifyTensorWithFixedData(outputTiles[0], 0);
        }
    }

    SECTION("DimNC tiling for weights, None for inputs") {
        // Inputs can all fit.
        TensorShape inputShape(
                { 1, 4096 }, DataLayout::NC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        fcOp->setInput(inputs, 0);
        // ...but weights can't. Weights will be tiled into 16 neuron-wise tiles
        // and 2 activation-wise tiles.
        fcOp->setNumOutputs(128);
        fcOp->createAllTensors();
        allocateAllTensors<float16>(fcOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(fcOp);
        REQUIRE(config.inputs == inputShape);
        REQUIRE(config.weights.dims() == std::vector<int>{ 8, 2048});
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 128 });

        SECTION("Generated tiles have correct shape and data") {
            fillTensorWithFixedData(inputs);
            TiledTensor inputTiles =
                    generateTiledTensorAndCopyData(inputs, config.inputs, fcOp);
            REQUIRE(inputTiles.size() == 1);
            verifyTensorWithFixedData(inputTiles[0], 0);

            auto weights = fcOp->getInput(1);
            fillTensorWithFixedData(weights);
            TiledTensor weightTiles = generateTiledTensorAndCopyData(
                    weights, config.weights, fcOp);
            REQUIRE(weightTiles.size() == 16 * 2);
            for (auto i = weightTiles.startIndex(); !i.end(); ++i) {
                REQUIRE(weightTiles[i]->getShape().dims() ==
                        std::vector<int>{ 8, 2048 });
                verifyTensorWithFixedData(weightTiles[i], (i % 2) * 2048);
            }

            auto outputs = fcOp->getOutput(0);
            fillTensorWithFixedData(outputs);
            TiledTensor outputTiles = generateTiledTensorAndCopyData(
                    outputs, config.outputs, fcOp);
            REQUIRE(outputTiles.size() == 1);
            verifyTensorWithFixedData(outputTiles[0], 0);
        }
    }

    SECTION("DimNC tiling for inputs and weights") {
        TensorShape inputShape(
                { 1, 32768 }, DataLayout::NC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        fcOp->setInput(inputs, 0);
        fcOp->setNumOutputs(256);
        fcOp->createAllTensors();
        allocateAllTensors<float16>(fcOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(fcOp);
        // Inputs/weights tiled into 16 activation-wise tiles, weights further
        // tiled into 32 neuron-wise tiles per activation-wise tile.
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 2048 });
        REQUIRE(config.weights.dims() == std::vector<int>{ 8, 2048 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 256 });

        SECTION("Generated tiles have correct shape and data") {
            fillTensorWithFixedData(inputs);
            TiledTensor inputTiles =
                    generateTiledTensorAndCopyData(inputs, config.inputs, fcOp);
            REQUIRE(inputTiles.size() == 16);
            for (auto i = inputTiles.startIndex(); !i.end(); ++i) {
                REQUIRE(inputTiles[i]->getShape().dims() ==
                        std::vector<int>{ 1, 2048 });
                verifyTensorWithFixedData(inputTiles[i], 2048 * i);
            }

            auto weights = fcOp->getInput(1);
            fillTensorWithFixedData(weights);
            TiledTensor weightTiles = generateTiledTensorAndCopyData(
                    weights, config.weights, fcOp);
            REQUIRE(weightTiles.size() == 16 * 32);
            for (auto i = weightTiles.startIndex(); !i.end(); ++i) {
                REQUIRE(weightTiles[i]->getShape().dims() ==
                        std::vector<int>{ 8, 2048 });
                verifyTensorWithFixedData(weightTiles[i], (i % 16) * 2048);
            }

            auto outputs = fcOp->getOutput(0);
            fillTensorWithFixedData(outputs);
            TiledTensor outputTiles = generateTiledTensorAndCopyData(
                    outputs, config.outputs, fcOp);
            REQUIRE(outputTiles.size() == 1);
            verifyTensorWithFixedData(outputTiles[0], 0);
        }
    }
}

