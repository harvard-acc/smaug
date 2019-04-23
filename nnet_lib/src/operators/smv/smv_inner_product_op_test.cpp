#include "catch.hpp"
#include "core/backend.h"
#include "core/tensor.h"
#include "core/smaug_test.h"
#include "operators/reorder_op.h"
#include "operators/smv/smv_test_common.h"
#include "operators/smv/smv_inner_product_op.h"
#include "operators/smv/smv_inner_product_tiling.h"

using namespace smaug;

Tensor* getReferenceOutput(SmvInnerProductOp* fcOp, Workspace* workspace) {
    auto input = fcOp->getInput(0);
    auto weights = fcOp->getInput(1);
    auto input32 = convertFp16ToFp32Tensor(input, workspace);
    auto weights32 = convertFp16ToFp32Tensor(weights, workspace);

    // Because we have transposed the weights, now we need to tranpose it back
    // for the reference implementation.
    auto transposedWeights32 = transpose2DTensor<float>(weights32);
    workspace->addTensor(transposedWeights32);

    // A reference inner product operator is used to get the 'correct' output.
    auto refFcOp = new InnerProductOp<ReferenceBackend>("ref_fc", workspace);
    refFcOp->setInput(input32, 0);
    refFcOp->setInput(transposedWeights32, 1);
    refFcOp->setNumOutputs(fcOp->getNumOutputs());
    refFcOp->createAllTensors();
    refFcOp->getOutput(0)->allocateStorage<float>();
    refFcOp->run();

    auto refOutput = refFcOp->getOutput(0);
    return convertFp32ToFp16Tensor(refOutput, workspace);
}

TEST_CASE_METHOD(SmaugTest, "SMV tiled inner product", "[smvfc]") {
    auto fcOp = new SmvInnerProductOp("fc", workspace());

    SECTION("No tiling required") {
        TensorShape inputShape(
                { 1, 256 }, DataLayout::NC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("input", inputShape);
        workspace()->addTensor(inputs);
        fcOp->setInput(inputs, 0);
        fcOp->setNumOutputs(32);
        inputs->allocateStorage<float16>();
        createAndFillTensorsWithData<float16>(fcOp, fillTensorWithData);
        fcOp->run();
        auto outputs = fcOp->getOutput(0);
        auto refOutputs = getReferenceOutput(fcOp, workspace());
        verifyOutputs<float16>(outputs, refOutputs);
    }

    SECTION("DimN tiling for weights, None for inputs") {
        TensorShape inputShape(
                { 1, 256 }, DataLayout::NC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        fcOp->setInput(inputs, 0);
        // Weights are tiled into 2 tiles.
        fcOp->setNumOutputs(128);
        createAndFillTensorsWithData<float16>(fcOp, fillTensorWithData);
        fcOp->run();
        auto outputs = fcOp->getOutput(0);
        auto refOutputs = getReferenceOutput(fcOp, workspace());
        verifyOutputs<float16>(outputs, refOutputs);
    }

    SECTION("DimNC tiling for weights, None for inputs") {
        TensorShape inputShape(
                { 1, 4096 }, DataLayout::NC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        fcOp->setInput(inputs, 0);
        // Weights are tiled into 16 neuron-wise tiles and 2 activation-wise
        // tiles.
        fcOp->setNumOutputs(128);
        createAndFillTensorsWithData<float16>(fcOp, fillTensorWithData);
        fcOp->run();
        auto outputs = fcOp->getOutput(0);
        auto refOutputs = getReferenceOutput(fcOp, workspace());
        verifyOutputs<float16>(outputs, refOutputs);
    }

    SECTION("DimNC tiling for weights and inputs") {
        TensorShape inputShape(
                { 1, 32768 }, DataLayout::NC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        fcOp->setInput(inputs, 0);
        // Inputs/weights are tiled into 16 activation-wise tiles, weights are
        // also tiled into 32 neuron-wise tiles.
        fcOp->setNumOutputs(256);
        createAndFillTensorsWithData<float16>(fcOp, fillTensorWithData);
        fcOp->run();
        auto outputs = fcOp->getOutput(0);
        auto refOutputs = getReferenceOutput(fcOp, workspace());
        verifyOutputs<float16>(outputs, refOutputs);
    }
}
