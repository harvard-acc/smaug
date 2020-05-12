#include "catch.hpp"
#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/core/smaug_test.h"
#include "smaug/operators/smv/smv_test_common.h"
#include "smaug/operators/smv/smv_batch_norm_op.h"
#include "smaug/operators/smv/smv_batch_norm_tiling.h"

using namespace smaug;

Tensor* concatWeightTensors(SmvBatchNormOp* bnOp) {
    auto mean = bnOp->getInput(SmvBatchNormOp::Mean);
    auto variance = bnOp->getInput(SmvBatchNormOp::Variance);
    auto gamma = bnOp->getInput(SmvBatchNormOp::Gamma);
    auto beta = bnOp->getInput(SmvBatchNormOp::Beta);
    // Concatenate the four weight tensors into one.
    return concatTensors(
            { mean, variance, gamma, beta }, 0, bnOp->getWorkspace());
}

TEST_CASE_METHOD(SmaugTest, "Post-conv bn tiling", "[smvtiling]") {
    using namespace smaug::smv;
    using namespace smaug::smv::bn;
    auto bnOp = new SmvBatchNormOp("bn", workspace());

    SECTION("No tiling needed") {
        TensorShape inputShape(
                { 1, 32, 32, 16 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        bnOp->setInput(inputs, 0);
        bnOp->createAllTensors();
        allocateAllTensors<float16>(bnOp);
        auto weights = concatWeightTensors(bnOp);
        auto outputs = bnOp->getOutput(0);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(
                inputs, weights, outputs);
        REQUIRE(config.inputs == inputShape);
        REQUIRE(config.weights.dims() == std::vector<int>{ 4, 16 });
        REQUIRE(config.outputs == inputShape);
    }

    SECTION("DimNC tiling") {
        TensorShape inputShape(
                { 1, 16, 16, 128 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        bnOp->setInput(inputs, 0);
        bnOp->createAllTensors();
        allocateAllTensors<float16>(bnOp);
        auto weights = concatWeightTensors(bnOp);
        auto outputs = bnOp->getOutput(0);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(
                inputs, weights, outputs);
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 16, 16, 64 });
        REQUIRE(config.weights == weights->getShape());
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 16, 16, 64 });

        SECTION("Generated tiles have correct shape and data") {
            fillTensorWithFixedData(inputs);
            TiledTensor inputsTiles =
                    generateTiledTensorAndCopyData(inputs, config.inputs, bnOp);
            REQUIRE(inputsTiles.size() == 2);
            for (auto i = inputsTiles.startIndex(); !i.end(); ++i) {
                REQUIRE(inputsTiles[i]->getShape().dims() ==
                        config.inputs.dims());
                verifyTensorWithFixedData(inputsTiles[i], 64 * i);
            }

            fillTensorWithFixedData(weights);
            TiledTensor weightsTiles = generateTiledTensorAndCopyData(
                    weights, config.weights, bnOp);
            REQUIRE(weightsTiles.size() == 1);
            REQUIRE(weightsTiles[0]->getShape().dims() ==
                    config.weights.dims());
            verifyTensorWithFixedData(weightsTiles[0], 0);

            fillTensorWithFixedData(outputs);
            TiledTensor outputsTiles = generateTiledTensorAndCopyData(
                    outputs, config.outputs, bnOp);
            REQUIRE(outputsTiles.size() == 2);
            for (auto i = outputsTiles.startIndex(); !i.end(); ++i) {
                REQUIRE(outputsTiles[i]->getShape().dims() ==
                        config.outputs.dims());
                verifyTensorWithFixedData(outputsTiles[i], 64 * i);
            }
        }
    }

    SECTION("DimNH tiling") {
        TensorShape inputShape(
                { 1, 64, 64, 32 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        bnOp->setInput(inputs, 0);
        bnOp->createAllTensors();
        allocateAllTensors<float16>(bnOp);
        auto weights = concatWeightTensors(bnOp);
        auto outputs = bnOp->getOutput(0);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(
                inputs, weights, outputs);
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 8, 64, 32 });
        REQUIRE(config.weights == weights->getShape());
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 8, 64, 32 });

        SECTION("Generated tiles have correct shape and data") {
            fillTensorWithFixedData(inputs);
            TiledTensor inputsTiles =
                    generateTiledTensorAndCopyData(inputs, config.inputs, bnOp);
            REQUIRE(inputsTiles.size() == 8);
            for (auto i = inputsTiles.startIndex(); !i.end(); ++i) {
                REQUIRE(inputsTiles[i]->getShape().dims() ==
                        config.inputs.dims());
                verifyTensorWithFixedData(inputsTiles[i], 0);
            }

            fillTensorWithFixedData(weights);
            TiledTensor weightsTiles = generateTiledTensorAndCopyData(
                    weights, config.weights, bnOp);
            REQUIRE(weightsTiles.size() == 1);
            REQUIRE(weightsTiles[0]->getShape().dims() ==
                    config.weights.dims());
            verifyTensorWithFixedData(weightsTiles[0], 0);

            fillTensorWithFixedData(outputs);
            TiledTensor outputsTiles = generateTiledTensorAndCopyData(
                    outputs, config.outputs, bnOp);
            REQUIRE(outputsTiles.size() == 8);
            for (auto i = outputsTiles.startIndex(); !i.end(); ++i) {
                REQUIRE(outputsTiles[i]->getShape().dims() ==
                        config.outputs.dims());
                verifyTensorWithFixedData(outputsTiles[i], 0);
            }
        }
    }

    SECTION("DimNW tiling") {
        TensorShape inputShape(
                { 1, 64, 1024, 32 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        bnOp->setInput(inputs, 0);
        bnOp->createAllTensors();
        allocateAllTensors<float16>(bnOp);
        auto weights = concatWeightTensors(bnOp);
        auto outputs = bnOp->getOutput(0);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(
                inputs, weights, outputs);
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 64, 8, 32 });
        REQUIRE(config.weights == weights->getShape());
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 64, 8, 32 });

        SECTION("Generated tiles have correct shape and data") {
            fillTensorWithFixedData(inputs);
            TiledTensor inputsTiles =
                    generateTiledTensorAndCopyData(inputs, config.inputs, bnOp);
            REQUIRE(inputsTiles.size() == 128);
            for (auto i = inputsTiles.startIndex(); !i.end(); ++i) {
                REQUIRE(inputsTiles[i]->getShape().dims() ==
                        config.inputs.dims());
                verifyTensorWithFixedData(inputsTiles[i], 0);
            }

            fillTensorWithFixedData(weights);
            TiledTensor weightsTiles = generateTiledTensorAndCopyData(
                    weights, config.weights, bnOp);
            REQUIRE(weightsTiles.size() == 1);
            REQUIRE(weightsTiles[0]->getShape().dims() ==
                    config.weights.dims());
            verifyTensorWithFixedData(weightsTiles[0], 0);

            fillTensorWithFixedData(outputs);
            TiledTensor outputsTiles = generateTiledTensorAndCopyData(
                    outputs, config.outputs, bnOp);
            REQUIRE(outputsTiles.size() == 128);
            for (auto i = outputsTiles.startIndex(); !i.end(); ++i) {
                REQUIRE(outputsTiles[i]->getShape().dims() ==
                        config.outputs.dims());
                verifyTensorWithFixedData(outputsTiles[i], 0);
            }
        }
    }

    SECTION("DimNHW tiling") {
        TensorShape inputShape(
                { 1, 128, 128, 64 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        bnOp->setInput(inputs, 0);
        bnOp->createAllTensors();
        allocateAllTensors<float16>(bnOp);
        auto weights = concatWeightTensors(bnOp);
        auto outputs = bnOp->getOutput(0);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(
                inputs, weights, outputs);
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 8, 32, 64 });
        REQUIRE(config.weights == weights->getShape());
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 8, 32, 64 });

        SECTION("Generated tiles have correct shape and data") {
            fillTensorWithFixedData(inputs);
            TiledTensor inputsTiles =
                    generateTiledTensorAndCopyData(inputs, config.inputs, bnOp);
            REQUIRE(inputsTiles.size() == 64);
            for (auto i = inputsTiles.startIndex(); !i.end(); ++i) {
                REQUIRE(inputsTiles[i]->getShape().dims() ==
                        config.inputs.dims());
                verifyTensorWithFixedData(inputsTiles[i], 0);
            }

            fillTensorWithFixedData(weights);
            TiledTensor weightsTiles = generateTiledTensorAndCopyData(
                    weights, config.weights, bnOp);
            REQUIRE(weightsTiles.size() == 1);
            REQUIRE(weightsTiles[0]->getShape().dims() ==
                    config.weights.dims());
            verifyTensorWithFixedData(weightsTiles[0], 0);

            fillTensorWithFixedData(outputs);
            TiledTensor outputsTiles = generateTiledTensorAndCopyData(
                    outputs, config.outputs, bnOp);
            REQUIRE(outputsTiles.size() == 64);
            for (auto i = outputsTiles.startIndex(); !i.end(); ++i) {
                REQUIRE(outputsTiles[i]->getShape().dims() ==
                        config.outputs.dims());
                verifyTensorWithFixedData(outputsTiles[i], 0);
            }
        }
    }

    SECTION("DimNCH tiling") {
        TensorShape inputShape(
                { 1, 64, 64, 512 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        bnOp->setInput(inputs, 0);
        bnOp->createAllTensors();
        allocateAllTensors<float16>(bnOp);
        auto weights = concatWeightTensors(bnOp);
        auto outputs = bnOp->getOutput(0);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(
                inputs, weights, outputs);
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 32, 64, 8 });
        REQUIRE(config.weights == weights->getShape());
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 32, 64, 8 });

        SECTION("Generated tiles have correct shape and data") {
            fillTensorWithFixedData(inputs);
            TiledTensor inputsTiles =
                    generateTiledTensorAndCopyData(inputs, config.inputs, bnOp);
            REQUIRE(inputsTiles.size() == 2 * 64);
            for (auto i = inputsTiles.startIndex(); !i.end(); ++i) {
                REQUIRE(inputsTiles[i]->getShape().dims() ==
                        config.inputs.dims());
                verifyTensorWithFixedData(inputsTiles[i], i % 64 * 8);
            }

            fillTensorWithFixedData(weights);
            TiledTensor weightsTiles = generateTiledTensorAndCopyData(
                    weights, config.weights, bnOp);
            REQUIRE(weightsTiles.size() == 1);
            REQUIRE(weightsTiles[0]->getShape().dims() ==
                    config.weights.dims());
            verifyTensorWithFixedData(weightsTiles[0], 0);

            fillTensorWithFixedData(outputs);
            TiledTensor outputsTiles = generateTiledTensorAndCopyData(
                    outputs, config.outputs, bnOp);
            REQUIRE(outputsTiles.size() == 2 * 64);
            for (auto i = outputsTiles.startIndex(); !i.end(); ++i) {
                REQUIRE(outputsTiles[i]->getShape().dims() ==
                        config.outputs.dims());
                verifyTensorWithFixedData(outputsTiles[i], i % 64 * 8);
            }
        }
    }

    SECTION("DimNCW tiling") {
        TensorShape inputShape(
                { 1, 64, 512, 512 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        bnOp->setInput(inputs, 0);
        bnOp->createAllTensors();
        allocateAllTensors<float16>(bnOp);
        auto weights = concatWeightTensors(bnOp);
        auto outputs = bnOp->getOutput(0);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(
                inputs, weights, outputs);
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 64, 32, 8 });
        REQUIRE(config.weights == weights->getShape());
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 64, 32, 8 });

        SECTION("Generated tiles have correct shape and data") {
            fillTensorWithFixedData(inputs);
            TiledTensor inputsTiles =
                    generateTiledTensorAndCopyData(inputs, config.inputs, bnOp);
            REQUIRE(inputsTiles.size() == 16 * 64);
            for (auto i = inputsTiles.startIndex(); !i.end(); ++i) {
                REQUIRE(inputsTiles[i]->getShape().dims() ==
                        config.inputs.dims());
                verifyTensorWithFixedData(inputsTiles[i], i % 64 * 8);
            }

            fillTensorWithFixedData(weights);
            TiledTensor weightsTiles = generateTiledTensorAndCopyData(
                    weights, config.weights, bnOp);
            REQUIRE(weightsTiles.size() == 1);
            REQUIRE(weightsTiles[0]->getShape().dims() ==
                    config.weights.dims());
            verifyTensorWithFixedData(weightsTiles[0], 0);

            fillTensorWithFixedData(outputs);
            TiledTensor outputsTiles = generateTiledTensorAndCopyData(
                    outputs, config.outputs, bnOp);
            REQUIRE(outputsTiles.size() == 16 * 64);
            for (auto i = outputsTiles.startIndex(); !i.end(); ++i) {
                REQUIRE(outputsTiles[i]->getShape().dims() ==
                        config.outputs.dims());
                verifyTensorWithFixedData(outputsTiles[i], i % 64 * 8);
            }
        }
    }
}

TEST_CASE_METHOD(SmaugTest, "Post-fc bn tiling", "[smvtiling]") {
    using namespace smaug::smv;
    using namespace smaug::smv::bn;
    auto bnOp = new SmvBatchNormOp("bn", workspace());

    SECTION("No tiling needed") {
        TensorShape inputShape(
                { 1, 1024 }, DataLayout::NC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        bnOp->setInput(inputs, 0);
        bnOp->createAllTensors();
        allocateAllTensors<float16>(bnOp);
        auto weights = concatWeightTensors(bnOp);
        auto outputs = bnOp->getOutput(0);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(
                inputs, weights, outputs);
        REQUIRE(config.inputs == inputShape);
        REQUIRE(config.weights.dims() == std::vector<int>{ 4, 1024 });
        REQUIRE(config.outputs == inputShape);
    }

    SECTION("DimNC tiling") {
        TensorShape inputShape(
                { 1, 32768 }, DataLayout::NC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        bnOp->setInput(inputs, 0);
        bnOp->createAllTensors();
        allocateAllTensors<float16>(bnOp);
        auto weights = concatWeightTensors(bnOp);
        auto outputs = bnOp->getOutput(0);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(
                inputs, weights, outputs);
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 4096 });
        REQUIRE(config.weights.dims() == std::vector<int>{ 4, 4096 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 4096 });

        SECTION("Generated tiles have correct shape and data") {
            fillTensorWithFixedData(inputs);
            TiledTensor inputsTiles =
                    generateTiledTensorAndCopyData(inputs, config.inputs, bnOp);
            REQUIRE(inputsTiles.size() == 8);
            for (auto i = inputsTiles.startIndex(); !i.end(); ++i) {
                REQUIRE(inputsTiles[i]->getShape().dims() ==
                        config.inputs.dims());
                verifyTensorWithFixedData(inputsTiles[i], 4096 * i);
            }

            fillTensorWithFixedData(weights);
            TiledTensor weightsTiles = generateTiledTensorAndCopyData(
                    weights, config.weights, bnOp);
            REQUIRE(weightsTiles.size() == 8);
            for (auto i = weightsTiles.startIndex(); !i.end(); ++i) {
                REQUIRE(weightsTiles[i]->getShape().dims() ==
                        config.weights.dims());
                verifyTensorWithFixedData(weightsTiles[i], 4096 * i);
            }

            fillTensorWithFixedData(outputs);
            TiledTensor outputsTiles = generateTiledTensorAndCopyData(
                    outputs, config.outputs, bnOp);
            REQUIRE(outputsTiles.size() == 8);
            for (auto i = outputsTiles.startIndex(); !i.end(); ++i) {
                REQUIRE(outputsTiles[i]->getShape().dims() ==
                        config.outputs.dims());
                verifyTensorWithFixedData(outputsTiles[i], 4096 * i);
            }
        }
    }
}
