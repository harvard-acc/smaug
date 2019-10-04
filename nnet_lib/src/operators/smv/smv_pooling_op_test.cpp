#include "catch.hpp"
#include "core/backend.h"
#include "core/tensor.h"
#include "core/smaug_test.h"
#include "operators/smv/smv_test_common.h"
#include "operators/smv/smv_pooling_op.h"
#include "operators/smv/smv_pooling_tiling.h"

using namespace smaug;

Tensor* getReferenceOutput(PoolingOp<SmvBackend>* poolOp,
                           Workspace* workspace) {
    OpType opType = poolOp->getOpType();
    auto input = poolOp->getInput(0);
    auto input32 = convertFp16ToFp32Tensor(input, workspace);

    // A reference pooling operator is used to get the 'correct' output.
    PoolingOp<ReferenceBackend>* refPoolOp;
    if (opType == MaxPooling)
        refPoolOp = new MaxPoolingOp<ReferenceBackend>("ref_pool", workspace);
    else
        refPoolOp = new AvgPoolingOp<ReferenceBackend>("ref_pool", workspace);
    refPoolOp->setPoolingSize(
            poolOp->getPoolingSize().first, poolOp->getPoolingSize().second);
    refPoolOp->setPoolingStride(poolOp->getPoolingStride().first,
                                poolOp->getPoolingStride().second);
    refPoolOp->setInput(input32, 0);
    refPoolOp->createAllTensors();
    refPoolOp->getOutput(0)->allocateStorage<float>();
    refPoolOp->run();
    return convertFp32ToFp16Tensor(refPoolOp->getOutput(0), workspace);
}

TEST_CASE_METHOD(SmaugTest, "SMV Tiled Pooling", "[smvpool]") {
    SECTION("Max pooling") {
        auto poolOp = new SmvMaxPoolingOp("pool", workspace());
        poolOp->setPoolingSize(2, 2);
        poolOp->setPoolingStride(2, 2);

        SECTION("No tiling required") {
            TensorShape inputShape(
                    { 1, 32, 32, 8 }, DataLayout::NHWC, SmvBackend::Alignment);
            Tensor* inputs = new Tensor("input", inputShape);
            inputs->allocateStorage<float16>();
            workspace()->addTensor(inputs);
            poolOp->setInput(inputs, 0);
            createAndFillTensorsWithData<float16>(
                    poolOp, fillTensorWithRandomData);
            poolOp->tile();
            poolOp->run();
            auto outputs = poolOp->getOutput(0);
            auto refOutputs = getReferenceOutput(poolOp, workspace());
            verifyOutputs<float16>(outputs, refOutputs);
        }

        SECTION("DimNC tiling on inputs, None for outputs") {
            // inputs tiled into 2 channelwise tiles.
            TensorShape inputShape(
                    { 1, 32, 32, 32 }, DataLayout::NHWC, SmvBackend::Alignment);
            Tensor* inputs = new Tensor("inputs", inputShape);
            inputs->allocateStorage<float16>();
            workspace()->addTensor(inputs);
            poolOp->setInput(inputs, 0);
            createAndFillTensorsWithData<float16>(
                    poolOp, fillTensorWithRandomData);
            poolOp->tile();
            poolOp->run();
            auto outputs = poolOp->getOutput(0);
            auto refOutputs = getReferenceOutput(poolOp, workspace());
            verifyOutputs<float16>(outputs, refOutputs);
        }

        SECTION("DimNC tiling on both inputs and outputs") {
            // inputs/outputs tiled into 8 channelwise tiles.
            TensorShape inputShape({ 1, 32, 32, 128 },
                                   DataLayout::NHWC,
                                   SmvBackend::Alignment);
            Tensor* inputs = new Tensor("inputs", inputShape);
            inputs->allocateStorage<float16>();
            workspace()->addTensor(inputs);
            poolOp->setInput(inputs, 0);
            createAndFillTensorsWithData<float16>(
                    poolOp, fillTensorWithRandomData);
            poolOp->tile();
            poolOp->run();
            auto outputs = poolOp->getOutput(0);
            auto refOutputs = getReferenceOutput(poolOp, workspace());
            verifyOutputs<float16>(outputs, refOutputs);
        }

        SECTION("DimNH tiling") {
            SECTION("Tiles have the same shape") {
                // Inputs/outputs tiled into 8 rowwise tiles.
                TensorShape inputShape({ 1, 64, 64, 32 },
                                       DataLayout::NHWC,
                                       SmvBackend::Alignment);
                Tensor* inputs = new Tensor("inputs", inputShape);
                inputs->allocateStorage<float16>();
                workspace()->addTensor(inputs);
                poolOp->setInput(inputs, 0);
                createAndFillTensorsWithData<float16>(
                        poolOp, fillTensorWithRandomData);
                poolOp->tile();
                poolOp->run();
                auto outputs = poolOp->getOutput(0);
                auto refOutputs = getReferenceOutput(poolOp, workspace());
                verifyOutputs<float16>(outputs, refOutputs);
            }
            SECTION("Last tile has a different shape") {
                // Inputs/outputs tiled into 12 rowwise tiles.
                TensorShape inputShape({ 1, 68, 68, 32 },
                                       DataLayout::NHWC,
                                       SmvBackend::Alignment);
                Tensor* inputs = new Tensor("inputs", inputShape);
                inputs->allocateStorage<float16>();
                workspace()->addTensor(inputs);
                poolOp->setInput(inputs, 0);
                createAndFillTensorsWithData<float16>(
                        poolOp, fillTensorWithRandomData);
                poolOp->tile();
                poolOp->run();
                auto outputs = poolOp->getOutput(0);
                auto refOutputs = getReferenceOutput(poolOp, workspace());
                verifyOutputs<float16>(outputs, refOutputs);
            }
        }

        SECTION("DimNCH tiling") {
            // In order to trigger DimNCH, we need big inputs.
            // Inputs and outputs are tiled into 256 rowwise tiles, inputs are
            // futher tiled into 2 channelwise tiles.
            TensorShape inputShape({ 1, 512, 1024, 16 },
                                   DataLayout::NHWC,
                                   SmvBackend::Alignment);
            Tensor* inputs = new Tensor("inputs", inputShape);
            inputs->allocateStorage<float16>();
            workspace()->addTensor(inputs);
            poolOp->setInput(inputs, 0);
            createAndFillTensorsWithData<float16>(
                    poolOp, fillTensorWithRandomData);
            poolOp->tile();
            poolOp->run();
            auto outputs = poolOp->getOutput(0);
            auto refOutputs = getReferenceOutput(poolOp, workspace());
            verifyOutputs<float16>(outputs, refOutputs);
        }
    }

    SECTION("Average pooling") {
        auto poolOp = new SmvAvgPoolingOp("pool", workspace());
        poolOp->setPoolingSize(2, 2);
        poolOp->setPoolingStride(2, 2);

        SECTION("No tiling required") {
            TensorShape inputShape(
                    { 1, 8, 8, 8 }, DataLayout::NHWC, SmvBackend::Alignment);
            Tensor* inputs = new Tensor("input", inputShape);
            inputs->allocateStorage<float16>();
            workspace()->addTensor(inputs);
            poolOp->setInput(inputs, 0);
            createAndFillTensorsWithData<float16>(
                    poolOp, fillTensorWithRandomData);
            poolOp->tile();
            poolOp->run();
            auto outputs = poolOp->getOutput(0);
            auto refOutputs = getReferenceOutput(poolOp, workspace());
            verifyOutputs<float16>(outputs, refOutputs);
        }

        SECTION("DimNC tiling on inputs, None for outputs") {
            // inputs tiled into 2 channelwise tiles.
            TensorShape inputShape(
                    { 1, 32, 32, 32 }, DataLayout::NHWC, SmvBackend::Alignment);
            Tensor* inputs = new Tensor("inputs", inputShape);
            inputs->allocateStorage<float16>();
            workspace()->addTensor(inputs);
            poolOp->setInput(inputs, 0);
            createAndFillTensorsWithData<float16>(
                    poolOp, fillTensorWithRandomData);
            poolOp->tile();
            poolOp->run();
            auto outputs = poolOp->getOutput(0);
            auto refOutputs = getReferenceOutput(poolOp, workspace());
            verifyOutputs<float16>(outputs, refOutputs);
        }

        SECTION("DimNC tiling on both inputs and outputs") {
            // inputs/outputs tiled into 8 channelwise tiles.
            TensorShape inputShape({ 1, 32, 32, 128 },
                                   DataLayout::NHWC,
                                   SmvBackend::Alignment);
            Tensor* inputs = new Tensor("inputs", inputShape);
            inputs->allocateStorage<float16>();
            workspace()->addTensor(inputs);
            poolOp->setInput(inputs, 0);
            createAndFillTensorsWithData<float16>(
                    poolOp, fillTensorWithRandomData);
            poolOp->tile();
            poolOp->run();
            auto outputs = poolOp->getOutput(0);
            auto refOutputs = getReferenceOutput(poolOp, workspace());
            verifyOutputs<float16>(outputs, refOutputs);
        }

        SECTION("DimNH tiling") {
            SECTION("Tiles have the same shape") {
                // Inputs/outputs tiled into 8 rowwise tiles.
                TensorShape inputShape({ 1, 64, 64, 32 },
                                       DataLayout::NHWC,
                                       SmvBackend::Alignment);
                Tensor* inputs = new Tensor("inputs", inputShape);
                inputs->allocateStorage<float16>();
                workspace()->addTensor(inputs);
                poolOp->setInput(inputs, 0);
                createAndFillTensorsWithData<float16>(
                        poolOp, fillTensorWithRandomData);
                poolOp->tile();
                poolOp->run();
                auto outputs = poolOp->getOutput(0);
                auto refOutputs = getReferenceOutput(poolOp, workspace());
                verifyOutputs<float16>(outputs, refOutputs);
            }
            SECTION("Last tile has a different shape") {
                // Inputs/outputs tiled into 12 rowwise tiles.
                TensorShape inputShape({ 1, 68, 68, 32 },
                                       DataLayout::NHWC,
                                       SmvBackend::Alignment);
                Tensor* inputs = new Tensor("inputs", inputShape);
                inputs->allocateStorage<float16>();
                workspace()->addTensor(inputs);
                poolOp->setInput(inputs, 0);
                createAndFillTensorsWithData<float16>(
                        poolOp, fillTensorWithRandomData);
                poolOp->tile();
                poolOp->run();
                auto outputs = poolOp->getOutput(0);
                auto refOutputs = getReferenceOutput(poolOp, workspace());
                verifyOutputs<float16>(outputs, refOutputs);
            }
        }

        SECTION("DimNCH tiling") {
            // In order to trigger DimNCH, we need big inputs.
            // Inputs and outputs are tiled into 256 rowwise tiles, inputs are
            // futher tiled into 2 channelwise tiles.
            TensorShape inputShape({ 1, 512, 1024, 16 },
                                   DataLayout::NHWC,
                                   SmvBackend::Alignment);
            Tensor* inputs = new Tensor("inputs", inputShape);
            inputs->allocateStorage<float16>();
            workspace()->addTensor(inputs);
            poolOp->setInput(inputs, 0);
            createAndFillTensorsWithData<float16>(
                    poolOp, fillTensorWithRandomData);
            poolOp->tile();
            poolOp->run();
            auto outputs = poolOp->getOutput(0);
            auto refOutputs = getReferenceOutput(poolOp, workspace());
            verifyOutputs<float16>(outputs, refOutputs);
        }
    }
}

