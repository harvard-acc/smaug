#include "catch.hpp"
#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/core/smaug_test.h"
#include "smaug/operators/smv/smv_test_common.h"
#include "smaug/operators/smv/smv_relu_op.h"
#include "smaug/operators/smv/smv_unary_op_common.h"

using namespace smaug;

void fillTensorWithIncrementData(Tensor* tensor) {
    const TensorShape& shape = tensor->getShape();
    float16* dataPtr = tensor->data<float16>();
    for (int i = 0; i < shape.storageSize(); i++) {
        dataPtr[i] = fp16(i);
    }
}

void verifyTensorWithIncrementData(Tensor* tensor, int valueOffset) {
    const TensorShape& shape = tensor->getShape();
    float16* dataPtr = tensor->data<float16>();
    for (int i = 0; i < shape.storageSize(); i++) {
        REQUIRE(Approx(fp32(dataPtr[i])).epsilon(kEpsilon) == i + valueOffset);
    }
}

TEST_CASE_METHOD(SmaugTest, "SMV 4D Unary tiling tests", "[smvtiling]") {
    using namespace smaug::smv;
    using namespace smaug::smv::unary;
    auto unaryOp = new SmvReluOp("relu", workspace());
    std::vector<int> tileDims = { 1, 16384 };

    SECTION("No tiling needed") {
        TensorShape inputShape(
                { 1, 32, 32, 8 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        unaryOp->setInput(inputs, 0);
        unaryOp->createAllTensors();
        allocateAllTensors<float16>(unaryOp);
        auto outputs = unaryOp->getOutput(0);
        std::array<TiledTensor, 2> tiledTensors = doTiling(unaryOp);
        TiledTensor& inputTiles = tiledTensors[0];
        TiledTensor& outputTiles = tiledTensors[1];
        REQUIRE(inputTiles.size() == 1);
        REQUIRE(inputTiles[0]->getShape().dims() ==
                std::vector<int>{ 1, 1 * 32 * 32 * 8 });
        REQUIRE(outputTiles.size() == 1);
        REQUIRE(outputTiles[0]->getShape().dims() ==
                std::vector<int>{ 1, 1 * 32 * 32 * 8 });
    }

    SECTION("Tiling needed, last tile has the same shape") {
        TensorShape inputShape(
                { 2, 16, 32, 32 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        unaryOp->setInput(inputs, 0);
        unaryOp->createAllTensors();
        allocateAllTensors<float16>(unaryOp);
        auto outputs = unaryOp->getOutput(0);
        fillTensorWithIncrementData(inputs);
        fillTensorWithIncrementData(outputs);
        std::array<TiledTensor, 2> tiledTensors = doTiling(unaryOp);
        TiledTensor& inputTiles = tiledTensors[0];
        TiledTensor& outputTiles = tiledTensors[1];

        SECTION("Generated tiles have correct shape and data") {
            REQUIRE(inputTiles.size() == 2);
            for (auto i = inputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = inputTiles[i]->getShape().dims();
                REQUIRE(testDims == tileDims);
                verifyTensorWithIncrementData(inputTiles[i], 16384 * i);
            }

            REQUIRE(outputTiles.size() == 2);
            for (auto i = outputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = outputTiles[i]->getShape().dims();
                REQUIRE(testDims == tileDims);
                verifyTensorWithIncrementData(outputTiles[i], 16384 * i);
            }
        }
    }

    SECTION("Tiling needed, last tile has a different shape") {
        TensorShape inputShape(
                { 2, 16, 32, 24 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        unaryOp->setInput(inputs, 0);
        unaryOp->createAllTensors();
        allocateAllTensors<float16>(unaryOp);
        auto outputs = unaryOp->getOutput(0);
        fillTensorWithIncrementData(inputs);
        fillTensorWithIncrementData(outputs);
        std::array<TiledTensor, 2> tiledTensors = doTiling(unaryOp);
        TiledTensor& inputTiles = tiledTensors[0];
        TiledTensor& outputTiles = tiledTensors[1];

        SECTION("Generated tiles have correct shape and data") {
            REQUIRE(inputTiles.size() == 2);
            for (auto i = inputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = inputTiles[i]->getShape().dims();
                if (i == 0 )
                    REQUIRE(testDims == tileDims);
                else if (i == 1)
                    REQUIRE(testDims == std::vector<int>{ 1, 8192 });
                verifyTensorWithIncrementData(inputTiles[i], 16384 * i);
            }

            REQUIRE(outputTiles.size() == 2);
            for (auto i = outputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = outputTiles[i]->getShape().dims();
                if (i == 0 )
                    REQUIRE(testDims == tileDims);
                else if (i == 1)
                    REQUIRE(testDims == std::vector<int>{ 1, 8192 });
                verifyTensorWithIncrementData(outputTiles[i], 16384 * i);
            }
        }
    }
}

TEST_CASE_METHOD(SmaugTest, "SMV 2D Unary tiling tests", "[smvtiling]") {
    using namespace smaug::smv;
    using namespace smaug::smv::unary;
    auto unaryOp = new SmvReluOp("relu", workspace());
    std::vector<int> tileDims = { 1, 16384 };

    SECTION("No tiling needed") {
        TensorShape inputShape(
                { 1, 1024 }, DataLayout::NC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        unaryOp->setInput(inputs, 0);
        unaryOp->createAllTensors();
        allocateAllTensors<float16>(unaryOp);
        auto outputs = unaryOp->getOutput(0);
        std::array<TiledTensor, 2> tiledTensors = doTiling(unaryOp);
        TiledTensor& inputTiles = tiledTensors[0];
        TiledTensor& outputTiles = tiledTensors[1];
        REQUIRE(inputTiles.size() == 1);
        REQUIRE(inputTiles[0]->getShape().dims() ==
                std::vector<int>{ 1, 1024 });
        REQUIRE(outputTiles.size() == 1);
        REQUIRE(outputTiles[0]->getShape().dims() ==
                std::vector<int>{ 1, 1024 });
    }

    SECTION("Tiling needed, last tile has the same shape") {
        TensorShape inputShape(
                { 2, 16384 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        unaryOp->setInput(inputs, 0);
        unaryOp->createAllTensors();
        allocateAllTensors<float16>(unaryOp);
        auto outputs = unaryOp->getOutput(0);
        fillTensorWithIncrementData(inputs);
        fillTensorWithIncrementData(outputs);
        std::array<TiledTensor, 2> tiledTensors = doTiling(unaryOp);
        TiledTensor& inputTiles = tiledTensors[0];
        TiledTensor& outputTiles = tiledTensors[1];

        SECTION("Generated tiles have correct shape and data") {
            REQUIRE(inputTiles.size() == 2);
            for (auto i = inputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = inputTiles[i]->getShape().dims();
                REQUIRE(testDims == tileDims);
                verifyTensorWithIncrementData(inputTiles[i], 16384 * i);
            }

            REQUIRE(outputTiles.size() == 2);
            for (auto i = outputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = outputTiles[i]->getShape().dims();
                REQUIRE(testDims == tileDims);
                verifyTensorWithIncrementData(outputTiles[i], 16384 * i);
            }
        }
    }

    SECTION("Tiling needed, last tile has a different shape") {
        TensorShape inputShape(
                { 2, 12288 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        unaryOp->setInput(inputs, 0);
        unaryOp->createAllTensors();
        allocateAllTensors<float16>(unaryOp);
        auto outputs = unaryOp->getOutput(0);
        fillTensorWithIncrementData(inputs);
        fillTensorWithIncrementData(outputs);
        std::array<TiledTensor, 2> tiledTensors = doTiling(unaryOp);
        TiledTensor& inputTiles = tiledTensors[0];
        TiledTensor& outputTiles = tiledTensors[1];

        SECTION("Generated tiles have correct shape and data") {
            REQUIRE(inputTiles.size() == 2);
            for (auto i = inputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = inputTiles[i]->getShape().dims();
                if (i == 0 )
                    REQUIRE(testDims == tileDims);
                else if (i == 1)
                    REQUIRE(testDims == std::vector<int>{ 1, 8192 });
                verifyTensorWithIncrementData(inputTiles[i], 16384 * i);
            }

            REQUIRE(outputTiles.size() == 2);
            for (auto i = outputTiles.startIndex(); !i.end(); ++i) {
                auto& testDims = outputTiles[i]->getShape().dims();
                if (i == 0 )
                    REQUIRE(testDims == tileDims);
                else if (i == 1)
                    REQUIRE(testDims == std::vector<int>{ 1, 8192 });
                verifyTensorWithIncrementData(outputTiles[i], 16384 * i);
            }
        }
    }
}

