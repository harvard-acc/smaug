#include "catch.hpp"
#include "core/backend.h"
#include "core/tensor.h"
#include "core/smaug_test.h"
#include "operators/reorder_op.h"
#include "operators/smv/smv_convolution_op.h"
#include "operators/smv/smv_convolution_tiling.h"

using namespace smaug;
using namespace smaug::smv::conv;

void fillTensorWithData(Tensor* tensor) {
    const TensorShape& shape = tensor->getShape();
    // Each dimension C is initialized to a different constant value.
    float* dataPtr = tensor->data<float>();
    int resetCounter = shape.getStorageDim(3);
    int value = 0;
    for (int i = 0; i < shape.storageSize(); i++) {
        dataPtr[i] = value++;
        if ((i + 1) % resetCounter == 0)
            value = 0;
    }
}

Tensor* getReferenceOutput(SmvConvolutionOp* convOp, Workspace* workspace) {
    auto input = convOp->getInput(0);
    auto kernels = convOp->getInput(1);

    // Two reorder operators for transforming input and kernels from NHWC to
    // NCHW.
    auto inputReorderOp =
            new ReorderOp<ReferenceBackend>("input/reorder", workspace);
    auto kernelReorderOp =
            new ReorderOp<ReferenceBackend>("kernel/reorder", workspace);
    inputReorderOp->setTargetLayout(NCHW);
    kernelReorderOp->setTargetLayout(NCHW);
    inputReorderOp->setInput(input, 0);
    kernelReorderOp->setInput(kernels, 0);
    inputReorderOp->createAllTensors();
    kernelReorderOp->createAllTensors();
    inputReorderOp->getOutput(0)->allocateStorage<float>();
    kernelReorderOp->getOutput(0)->allocateStorage<float>();
    inputReorderOp->run();
    kernelReorderOp->run();
    auto refInput = inputReorderOp->getOutput(0);
    auto refKernels = kernelReorderOp->getOutput(0);

    // A reference convolution operator is used to get the 'correct' output.
    auto refConvOp = new ConvolutionOp<ReferenceBackend>("ref_conv", workspace);
    refConvOp->setPadding(convOp->getPadding());
    refConvOp->setWeightDims(kernels->getShape()[1], kernels->getShape()[2],
                             kernels->getShape()[0]);
    refConvOp->setStride(convOp->getRowStride(), convOp->getColStride());
    refConvOp->setInput(refInput, 0);
    refConvOp->setInput(refKernels, 1);
    refConvOp->createAllTensors();
    refConvOp->getOutput(0)->allocateStorage<float>();
    refConvOp->run();

    // The output of the reference convolution operator needs to be tranformed
    // back to NHWC for verification later.
    auto refOutput = refConvOp->getOutput(0);
    auto outputReorderOp =
            new ReorderOp<ReferenceBackend>("output/reorder", workspace);
    outputReorderOp->setTargetLayout(NHWC);
    outputReorderOp->setInput(refOutput, 0);
    outputReorderOp->createAllTensors();
    outputReorderOp->getOutput(0)->allocateStorage<float>();
    outputReorderOp->run();
    return outputReorderOp->getOutput(0);
}

TEST_CASE_METHOD(SmaugTest, "SMV Tiled Convolution", "[smvconv]") {
    smv::kSpadSize = 32 * 1024;
    auto convOp = new SmvConvolutionOp("conv", workspace());
    // Outputs should be the same size as inputs.
    convOp->setStride(1, 1);
    convOp->setPadding(SamePadding);

    SECTION("No tiling required") {
        TensorShape inputShape(
                { 1, 8, 8, 8 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("input", inputShape);
        inputs->allocateStorage<float>();
        workspace()->addTensor(inputs);
        convOp->setInput(inputs, 0);
        convOp->setWeightDims(3, 3, 8);
        SECTION("Same padding") {
            convOp->createAllTensors();
            allocateAllTensors<float>(convOp);
            fillTensorWithData(inputs);
            fillTensorWithData(convOp->getInput(1));
            convOp->run();
            auto outputs = convOp->getOutput(0);
            auto refOutputs = getReferenceOutput(convOp, workspace());
            verifyOutputs<float>(outputs, refOutputs);
        }
        SECTION("Valid padding") {
            convOp->setPadding(ValidPadding);
            convOp->createAllTensors();
            allocateAllTensors<float>(convOp);
            fillTensorWithData(inputs);
            fillTensorWithData(convOp->getInput(1));
            convOp->run();
            auto outputs = convOp->getOutput(0);
            auto refOutputs = getReferenceOutput(convOp, workspace());
            verifyOutputs<float>(outputs, refOutputs);
        }
    }

    SECTION("DimN tiled convolution") {
        SECTION("Every weight tile contains 8 kernels") {
            TensorShape inputShape(
                    { 1, 8, 8, 96 }, DataLayout::NHWC, SmvBackend::Alignment);
            Tensor* inputs = new Tensor("inputs", inputShape);
            workspace()->addTensor(inputs);
            convOp->setInput(inputs, 0);
            convOp->setWeightDims(3, 3, 128);
            convOp->createAllTensors();
            allocateAllTensors<float>(convOp);
            fillTensorWithData(inputs);
            fillTensorWithData(convOp->getInput(1));
            convOp->run();
            auto outputs = convOp->getOutput(0);
            auto refOutputs = getReferenceOutput(convOp, workspace());
            verifyOutputs<float>(outputs, refOutputs);
        }
        SECTION("Every weight tile contains more than 8 kernels") {
            TensorShape inputShape(
                    { 1, 8, 8, 16 }, DataLayout::NHWC, SmvBackend::Alignment);
            Tensor* inputs = new Tensor("inputs", inputShape);
            workspace()->addTensor(inputs);
            convOp->setInput(inputs, 0);
            // The weight tiles will contain 56, 56 and 16 kernels respectively.
            convOp->setWeightDims(3, 3, 128);
            convOp->createAllTensors();
            allocateAllTensors<float>(convOp);
            fillTensorWithData(inputs);
            fillTensorWithData(convOp->getInput(1));
            convOp->run();
            auto outputs = convOp->getOutput(0);
            auto refOutputs = getReferenceOutput(convOp, workspace());
            verifyOutputs<float>(outputs, refOutputs);
        }
    }

    SECTION("DimNH tiled convolution") {
        TensorShape inputShape(
                { 1, 32, 32, 16 }, DataLayout::NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("inputs", inputShape);
        workspace()->addTensor(inputs);
        convOp->setInput(inputs, 0);
        convOp->setWeightDims(3, 3, 8);
        convOp->createAllTensors();
        allocateAllTensors<float>(convOp);
        fillTensorWithData(inputs);
        fillTensorWithData(convOp->getInput(1));
        convOp->run();
        auto outputs = convOp->getOutput(0);
        auto refOutputs = getReferenceOutput(convOp, workspace());
        verifyOutputs<float>(outputs, refOutputs);
    }

    SECTION("DimNC tiled convolution") {
        SECTION("Input tile and weight tile have the same channel dimension."
                "Both are 1") {
            TensorShape inputShape(
                    { 1, 16, 8, 128 }, DataLayout::NHWC, SmvBackend::Alignment);
            Tensor* inputs = new Tensor("inputs", inputShape);
            workspace()->addTensor(inputs);
            convOp->setInput(inputs, 0);
            convOp->setWeightDims(3, 3, 8);
            convOp->createAllTensors();
            allocateAllTensors<float>(convOp);
            fillTensorWithData(inputs);
            fillTensorWithData(convOp->getInput(1));
            convOp->run();
            auto outputs = convOp->getOutput(0);
            auto refOutputs = getReferenceOutput(convOp, workspace());
            verifyOutputs<float>(outputs, refOutputs);
        }
        SECTION("Inputs are not tiled channelwise, weights have 2 channelwise "
                "tiles") {
            TensorShape inputShape(
                    { 1, 8, 8, 128 }, DataLayout::NHWC, SmvBackend::Alignment);
            Tensor* inputs = new Tensor("inputs", inputShape);
            workspace()->addTensor(inputs);
            convOp->setInput(inputs, 0);
            convOp->setWeightDims(3, 3, 8);
            convOp->createAllTensors();
            allocateAllTensors<float>(convOp);
            fillTensorWithData(inputs);
            fillTensorWithData(convOp->getInput(1));
            convOp->run();
            auto outputs = convOp->getOutput(0);
            auto refOutputs = getReferenceOutput(convOp, workspace());
            verifyOutputs<float>(outputs, refOutputs);
        }
    }
}
