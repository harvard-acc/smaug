#include "catch.hpp"
#include "core/backend.h"
#include "core/tensor.h"
#include "core/smaug_test.h"
#include "operators/smv/smv_test_common.h"
#include "operators/smv/smv_pooling_op.h"
#include "operators/smv/smv_pooling_tiling.h"

using namespace smaug;

TEST_CASE_METHOD(SmaugTest, "SMV Pooling tiling tests", "[smvtiling]") {
    using namespace smaug::smv;
    using namespace smaug::smv::pool;
    auto poolOp = new SmvMaxPoolingOp("pool", workspace());
    poolOp->setPoolingSize(2, 2);
    poolOp->setPoolingStride(2, 2);

    SECTION("No tiling needed") {
        TensorShape inputShape(
                { 1, 32, 32, 8 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        poolOp->setInput(inputs, 0);
        poolOp->createAllTensors();
        allocateAllTensors<float16>(poolOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(poolOp);
        REQUIRE(config.inputs == inputShape);
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 16, 16, 8 });
    }

    SECTION("DimNC tiling on inputs, None for outputs") {
        // inputs tiled into 2 channelwise tiles.
        TensorShape inputShape(
                { 1, 32, 32, 32 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        poolOp->setInput(inputs, 0);
        poolOp->createAllTensors();
        allocateAllTensors<float16>(poolOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(poolOp);
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 32, 32, 16 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 16, 16, 32 });

        SECTION("Generated tiles have correct shape and data") {
            fillTensorWithFixedData(inputs);
            TiledTensor inputTiles = generateTiledTensor(
                    inputs, config.inputs, { 0, 0, 0, 0 }, poolOp);
            REQUIRE(inputTiles.size() == 2);
            for (auto i = inputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = inputTiles[i]->getShape().dims();
                REQUIRE(testDims == config.inputs.dims());
                verifyTensorWithFixedData(inputTiles[i], 16 * i);
            }

            auto outputs = poolOp->getOutput(0);
            fillTensorWithFixedData(outputs);
            TiledTensor outputTiles = generateTiledTensor(
                    outputs, config.outputs, { 0, 0, 0, 0 }, poolOp);
            REQUIRE(outputTiles.size() == 1);
            REQUIRE(outputTiles[0]->getShape().dims() == config.outputs.dims());
            verifyTensorWithFixedData(outputTiles[0], 0);
        }
    }

    SECTION("DimNC tiling on both inputs and outputs") {
        // inputs/outputs tiled into 8 channelwise tiles.
        TensorShape inputShape(
                { 1, 32, 32, 128 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        poolOp->setInput(inputs, 0);
        poolOp->createAllTensors();
        allocateAllTensors<float16>(poolOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(poolOp);
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 32, 32, 16 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 16, 16, 16 });

        SECTION("Generated tiles have correct shape and data") {
            fillTensorWithFixedData(inputs);
            TiledTensor inputTiles = generateTiledTensor(
                    inputs, config.inputs, { 0, 0, 0, 0 }, poolOp);
            REQUIRE(inputTiles.size() == 8);
            for (auto i = inputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = inputTiles[i]->getShape().dims();
                REQUIRE(testDims == config.inputs.dims());
                verifyTensorWithFixedData(inputTiles[i], 16 * i);
            }

            auto outputs = poolOp->getOutput(0);
            fillTensorWithFixedData(outputs);
            TiledTensor outputTiles = generateTiledTensor(
                    outputs, config.outputs, { 0, 0, 0, 0 }, poolOp);
            REQUIRE(outputTiles.size() == 8);
            for (auto i = outputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = outputTiles[i]->getShape().dims();
                REQUIRE(testDims == config.outputs.dims());
                verifyTensorWithFixedData(outputTiles[i], 16 * i);
            }
        }
    }

    SECTION("DimNH tiling") {
        SECTION("Tiles have the same shape") {
            // Inputs/outputs tiled into 8 rowwise tiles.
            TensorShape inputShape(
                    { 1, 64, 64, 32 }, DataLayout::NHWC, SmvBackend::Alignment);
            Tensor* inputs = new Tensor("inputs", inputShape);
            workspace()->addTensor(inputs);
            poolOp->setInput(inputs, 0);
            poolOp->createAllTensors();
            allocateAllTensors<float16>(poolOp);
            TilingConfig config =
                    TilingOptimizer::computeBasicTileShapes(poolOp);
            REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 8, 64, 32 });
            REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 4, 32, 32 });

            SECTION("Generated tiles have correct shape and data") {
                fillTensorWithFixedData(inputs);
                TiledTensor inputTiles = generateTiledTensor(
                        inputs, config.inputs, { 0, 0, 0, 0 }, poolOp);
                REQUIRE(inputTiles.size() == 8);
                for (auto i = inputTiles.startIndex(); !i.end(); ++i)
                    verifyTensorWithFixedData(inputTiles[i], 0);

                auto outputs = poolOp->getOutput(0);
                fillTensorWithFixedData(outputs);
                TiledTensor outputTiles = generateTiledTensor(
                        outputs, config.outputs, { 0, 0, 0, 0 }, poolOp);
                REQUIRE(outputTiles.size() == 8);
                for (auto i = outputTiles.startIndex(); !i.end(); ++i)
                    verifyTensorWithFixedData(outputTiles[i], 0);
            }
        }
        SECTION("Last tile has a different shape") {
            // Inputs/outputs tiled into 12 rowwise tiles.
            TensorShape inputShape(
                    { 1, 68, 68, 32 }, DataLayout::NHWC, SmvBackend::Alignment);
            Tensor* inputs = new Tensor("inputs", inputShape);
            workspace()->addTensor(inputs);
            poolOp->setInput(inputs, 0);
            poolOp->createAllTensors();
            allocateAllTensors<float16>(poolOp);
            TilingConfig config =
                    TilingOptimizer::computeBasicTileShapes(poolOp);
            REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 6, 68, 32 });
            REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 3, 34, 32 });

            SECTION("Generated tiles have correct shape and data") {
                fillTensorWithFixedData(inputs);
                TiledTensor inputTiles = generateTiledTensor(
                        inputs, config.inputs, { 0, 0, 0, 0 }, poolOp);
                REQUIRE(inputTiles.size() == 12);
                for (auto i = inputTiles.startIndex(); !i.end(); ++i) {
                    auto& testDims = inputTiles[i]->getShape().dims();
                    if (i < 11) {
                        REQUIRE(testDims == config.inputs.dims());
                    } else {
                        // Last tile.
                        REQUIRE(testDims == std::vector<int>{ 1, 2, 68, 32 });
                    }
                    verifyTensorWithFixedData(inputTiles[i], 0);
                }

                auto outputs = poolOp->getOutput(0);
                fillTensorWithFixedData(outputs);
                TiledTensor outputTiles = generateTiledTensor(
                        outputs, config.outputs, { 0, 0, 0, 0 }, poolOp);
                REQUIRE(outputTiles.size() == 12);
                for (auto i = outputTiles.startIndex(); !i.end(); ++i) {
                    auto& testDims = outputTiles[i]->getShape().dims();
                    if (i < 11) {
                        REQUIRE(testDims == config.outputs.dims());
                    } else {
                        // Last tile.
                        REQUIRE(testDims == std::vector<int>{ 1, 1, 34, 32 });
                    }
                    verifyTensorWithFixedData(outputTiles[i], 0);
                }
            }
        }
    }

    SECTION("DimNCH tiling") {
        // In order to trigger DimNCH, we need big inputs.
        // Inputs and outputs are tiled into 256 rowwise tiles, inputs are
        // futher tiled into 2 channelwise tiles.
        TensorShape inputShape(
                { 1, 512, 1024, 16 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        poolOp->setInput(inputs, 0);
        poolOp->createAllTensors();
        allocateAllTensors<float16>(poolOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(poolOp);
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 2, 1024, 8 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 1, 512, 16 });

        SECTION("Generated tiles have correct shape and data") {
            fillTensorWithFixedData(inputs);
            TiledTensor inputTiles = generateTiledTensor(
                    inputs, config.inputs, { 0, 0, 0, 0 }, poolOp);
            REQUIRE(inputTiles.size() == 256 * 2);
            for (auto i = inputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = inputTiles[i]->getShape().dims();
                REQUIRE(testDims == config.inputs.dims());
                verifyTensorWithFixedData(inputTiles[i], (i % 2) * 8);
            }

            auto outputs = poolOp->getOutput(0);
            fillTensorWithFixedData(outputs);
            TiledTensor outputTiles = generateTiledTensor(
                    outputs, config.outputs, { 0, 0, 0, 0 }, poolOp);
            REQUIRE(outputTiles.size() == 256);
            for (auto i = outputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = outputTiles[i]->getShape().dims();
                REQUIRE(testDims == config.outputs.dims());
                verifyTensorWithFixedData(outputTiles[i], 0);
            }
        }
    }
}

