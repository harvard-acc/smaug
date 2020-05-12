#include "catch.hpp"
#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/core/smaug_test.h"
#include "smaug/operators/smv/smv_test_common.h"
#include "smaug/operators/smv/smv_pooling_op.h"
#include "smaug/operators/smv/smv_pooling_tiling.h"

using namespace smaug;

TiledTensor generateInputTiles(SmvPoolingOp* poolOp,
                               const TensorShape& tileShape) {
    Tensor* inputs = poolOp->getInput(0);
    int poolRowSize, poolColSize, poolRowStride, poolColStride;
    std::tie(poolRowSize, poolColSize) = poolOp->getPoolingSize();
    std::tie(poolRowStride, poolColStride) = poolOp->getPoolingStride();
    return generateTiledTensorAndCopyData(inputs,
                                          tileShape,
                                          poolOp,
                                          poolRowSize,
                                          poolColSize,
                                          poolRowStride,
                                          poolColStride);
}

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
            TiledTensor inputTiles = generateInputTiles(poolOp, config.inputs);
            REQUIRE(inputTiles.size() == 2);
            for (auto i = inputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = inputTiles[i]->getShape().dims();
                REQUIRE(testDims == config.inputs.dims());
                verifyTensorWithFixedData(inputTiles[i], 16 * i);
            }

            auto outputs = poolOp->getOutput(0);
            fillTensorWithFixedData(outputs);
            TiledTensor outputTiles = generateTiledTensorAndCopyData(
                    outputs, config.outputs, poolOp);
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
            TiledTensor inputTiles = generateInputTiles(poolOp, config.inputs);
            REQUIRE(inputTiles.size() == 8);
            for (auto i = inputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = inputTiles[i]->getShape().dims();
                REQUIRE(testDims == config.inputs.dims());
                verifyTensorWithFixedData(inputTiles[i], 16 * i);
            }

            auto outputs = poolOp->getOutput(0);
            fillTensorWithFixedData(outputs);
            TiledTensor outputTiles = generateTiledTensorAndCopyData(
                    outputs, config.outputs, poolOp);
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
                TiledTensor inputTiles =
                        generateInputTiles(poolOp, config.inputs);
                REQUIRE(inputTiles.size() == 8);
                for (auto i = inputTiles.startIndex(); !i.end(); ++i)
                    verifyTensorWithFixedData(inputTiles[i], 0);

                auto outputs = poolOp->getOutput(0);
                fillTensorWithFixedData(outputs);
                TiledTensor outputTiles = generateTiledTensorAndCopyData(
                        outputs, config.outputs, poolOp);
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
                TiledTensor inputTiles =
                        generateInputTiles(poolOp, config.inputs);
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
                TiledTensor outputTiles = generateTiledTensorAndCopyData(
                        outputs, config.outputs, poolOp);
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

    SECTION("DimNW tiling") {
        // Inputs and outputs are tiled into 512 columnwise tiles.
        TensorShape inputShape(
                { 1, 512, 1024, 16 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        poolOp->setInput(inputs, 0);
        poolOp->createAllTensors();
        allocateAllTensors<float16>(poolOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(poolOp);
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 512, 2, 16 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 256, 1, 16 });

        SECTION("Generated tiles have correct shape and data") {
            fillTensorWithFixedData(inputs);
            TiledTensor inputTiles = generateInputTiles(poolOp, config.inputs);
            REQUIRE(inputTiles.size() == 512);
            for (auto i = inputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = inputTiles[i]->getShape().dims();
                REQUIRE(testDims == config.inputs.dims());
                verifyTensorWithFixedData(inputTiles[i], 0);
            }

            auto outputs = poolOp->getOutput(0);
            fillTensorWithFixedData(outputs);
            TiledTensor outputTiles = generateTiledTensorAndCopyData(
                    outputs, config.outputs, poolOp);
            REQUIRE(outputTiles.size() == 512);
            for (auto i = outputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = outputTiles[i]->getShape().dims();
                REQUIRE(testDims == config.outputs.dims());
                verifyTensorWithFixedData(outputTiles[i], 0);
            }
        }
    }

    SECTION("DimNHW tiling") {
        // Inputs and outputs are tiled into 256 rowwise tiles and 2 columnwise
        // tiles.
        TensorShape inputShape(
                { 1, 512, 512, 32 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        poolOp->setInput(inputs, 0);
        poolOp->createAllTensors();
        allocateAllTensors<float16>(poolOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(poolOp);
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 2, 256, 32 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 1, 128, 32 });

        SECTION("Generated tiles have correct shape and data") {
            fillTensorWithFixedData(inputs);
            TiledTensor inputTiles = generateInputTiles(poolOp, config.inputs);
            REQUIRE(inputTiles.size() == 512);
            for (auto i = inputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = inputTiles[i]->getShape().dims();
                REQUIRE(testDims == config.inputs.dims());
                verifyTensorWithFixedData(inputTiles[i], 0);
            }

            auto outputs = poolOp->getOutput(0);
            fillTensorWithFixedData(outputs);
            TiledTensor outputTiles = generateTiledTensorAndCopyData(
                    outputs, config.outputs, poolOp);
            REQUIRE(outputTiles.size() == 512);
            for (auto i = outputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = outputTiles[i]->getShape().dims();
                REQUIRE(testDims == config.outputs.dims());
                verifyTensorWithFixedData(outputTiles[i], 0);
            }
        }
    }

    SECTION("Channelwise tiling") {
        // To trigger channelwise tiling, we need big pooling sizes.
        // Realistically speaking, these pooling size are not used in real
        // models, but for testing purpose, let's use them.
        poolOp->setPoolingSize(16, 16);
        poolOp->setPoolingStride(16, 16);

        SECTION("DimNCH tiling") {
            // Inputs and outputs are tiled into 2 rowwise tiles, and input are
            // tiled into 16 channelwise tiles.
            TensorShape inputShape({ 1, 64, 64, 128 },
                                   DataLayout::NHWC,
                                   SmvBackend::Alignment);
            Tensor* inputs = new Tensor("inputs", inputShape);
            workspace()->addTensor(inputs);
            poolOp->setInput(inputs, 0);
            poolOp->createAllTensors();
            allocateAllTensors<float16>(poolOp);
            TilingConfig config =
                    TilingOptimizer::computeBasicTileShapes(poolOp);
            REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 32, 64, 8 });
            REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 2, 4, 128 });

            SECTION("Generated tiles have correct shape and data") {
                fillTensorWithFixedData(inputs);
                TiledTensor inputTiles =
                        generateInputTiles(poolOp, config.inputs);
                REQUIRE(inputTiles.size() == 32);
                for (auto i = inputTiles.startIndex(); !i.end(); ++i) {
                    auto& testDims = inputTiles[i]->getShape().dims();
                    REQUIRE(testDims == config.inputs.dims());
                    verifyTensorWithFixedData(inputTiles[i], (i % 16) * 8);
                }

                auto outputs = poolOp->getOutput(0);
                fillTensorWithFixedData(outputs);
                TiledTensor outputTiles = generateTiledTensorAndCopyData(
                        outputs, config.outputs, poolOp);
                REQUIRE(outputTiles.size() == 2);
                for (auto i = outputTiles.startIndex(); !i.end(); ++i) {
                    auto& testDims = outputTiles[i]->getShape().dims();
                    REQUIRE(testDims == config.outputs.dims());
                    verifyTensorWithFixedData(outputTiles[i], 0);
                }
            }
        }

        SECTION("DimNCW tiling") {
            // Inputs and outputs are tiled into 8 columnwise tiles, and input
            // are tiled into 16 channelwise tiles.
            TensorShape inputShape({ 1, 64, 256, 128 },
                                   DataLayout::NHWC,
                                   SmvBackend::Alignment);
            Tensor* inputs = new Tensor("inputs", inputShape);
            workspace()->addTensor(inputs);
            poolOp->setInput(inputs, 0);
            poolOp->createAllTensors();
            allocateAllTensors<float16>(poolOp);
            TilingConfig config =
                    TilingOptimizer::computeBasicTileShapes(poolOp);
            REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 64, 32, 8 });
            REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 4, 2, 128 });

            SECTION("Generated tiles have correct shape and data") {
                fillTensorWithFixedData(inputs);
                TiledTensor inputTiles =
                        generateInputTiles(poolOp, config.inputs);
                REQUIRE(inputTiles.size() == 128);
                for (auto i = inputTiles.startIndex(); !i.end(); ++i) {
                    auto& testDims = inputTiles[i]->getShape().dims();
                    REQUIRE(testDims == config.inputs.dims());
                    verifyTensorWithFixedData(inputTiles[i], (i % 16) * 8);
                }

                auto outputs = poolOp->getOutput(0);
                fillTensorWithFixedData(outputs);
                TiledTensor outputTiles = generateTiledTensorAndCopyData(
                        outputs, config.outputs, poolOp);
                REQUIRE(outputTiles.size() == 8);
                for (auto i = outputTiles.startIndex(); !i.end(); ++i) {
                    auto& testDims = outputTiles[i]->getShape().dims();
                    REQUIRE(testDims == config.outputs.dims());
                    verifyTensorWithFixedData(outputTiles[i], 0);
                }
            }
        }
    }
}

