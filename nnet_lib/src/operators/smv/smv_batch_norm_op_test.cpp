#include "catch.hpp"
#include "core/backend.h"
#include "core/tensor.h"
#include "core/smaug_test.h"
#include "operators/smv/smv_test_common.h"
#include "operators/smv/smv_batch_norm_op.h"

using namespace smaug;

Tensor* getReferenceOutput(SmvBatchNormOp* bnOp,
                           Workspace* workspace) {
    auto input = bnOp->getInput(0);
    auto mean = bnOp->getInput(SmvBatchNormOp::Mean);
    auto variance = bnOp->getInput(SmvBatchNormOp::Variance);
    auto gamma = bnOp->getInput(SmvBatchNormOp::Gamma);
    auto beta = bnOp->getInput(SmvBatchNormOp::Beta);
    auto input32 = convertFp16ToFp32Tensor(input, workspace);
    auto mean32 = convertFp16ToFp32Tensor(mean, workspace);
    auto variance32 = convertFp16ToFp32Tensor(variance, workspace);
    auto gamma32 = convertFp16ToFp32Tensor(gamma, workspace);
    auto beta32 = convertFp16ToFp32Tensor(beta, workspace);

    // A reference batch norm operator is used to get the 'correct' output.
    BatchNormOp<ReferenceBackend>* refBnOp =
            new BatchNormOp<ReferenceBackend>("ref_bn", workspace);
    refBnOp->setInput(input32, 0);
    refBnOp->setInput(mean32, SmvBatchNormOp::Mean);
    refBnOp->setInput(variance32, SmvBatchNormOp::Variance);
    refBnOp->setInput(gamma32, SmvBatchNormOp::Gamma);
    refBnOp->setInput(beta32, SmvBatchNormOp::Beta);
    refBnOp->createAllTensors();
    refBnOp->getOutput(0)->allocateStorage<float>();
    refBnOp->run();
    return convertFp32ToFp16Tensor(refBnOp->getOutput(0), workspace);
}

TEST_CASE_METHOD(SmaugTest, "SMV Post-Conv Batch Norm", "[smvpool]") {
    auto bnOp = new SmvBatchNormOp("bn", workspace());

    SECTION("No tiling required") {
        TensorShape inputShape(
                { 1, 16, 32, 32 }, DataLayout::NCHW, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("input", inputShape);
        inputs->allocateStorage<float16>();
        workspace()->addTensor(inputs);
        bnOp->setInput(inputs, 0);
        createAndFillTensorsWithData<float16>(bnOp, fillTensorWithRandomData);
        bnOp->run();
        auto outputs = bnOp->getOutput(0);
        auto refOutputs = getReferenceOutput(bnOp, workspace());
        verifyOutputs<float16>(outputs, refOutputs);
    }

    SECTION("DimNC tiling") {
        TensorShape inputShape(
                { 1, 128, 16, 16 }, DataLayout::NCHW, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("input", inputShape);
        inputs->allocateStorage<float16>();
        workspace()->addTensor(inputs);
        bnOp->setInput(inputs, 0);
        createAndFillTensorsWithData<float16>(bnOp, fillTensorWithRandomData);
        bnOp->run();
        auto outputs = bnOp->getOutput(0);
        auto refOutputs = getReferenceOutput(bnOp, workspace());
        verifyOutputs<float16>(outputs, refOutputs);
    }

    SECTION("DimNW tiling") {
        TensorShape inputShape(
                { 1, 32, 64, 64 }, DataLayout::NCHW, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("input", inputShape);
        inputs->allocateStorage<float16>();
        workspace()->addTensor(inputs);
        bnOp->setInput(inputs, 0);
        createAndFillTensorsWithData<float16>(bnOp, fillTensorWithRandomData);
        bnOp->run();
        auto outputs = bnOp->getOutput(0);
        auto refOutputs = getReferenceOutput(bnOp, workspace());
        verifyOutputs<float16>(outputs, refOutputs);
    }

    SECTION("DimNCW tiling") {
        TensorShape inputShape(
                { 1, 64, 128, 128 }, DataLayout::NCHW, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("input", inputShape);
        inputs->allocateStorage<float16>();
        workspace()->addTensor(inputs);
        bnOp->setInput(inputs, 0);
        createAndFillTensorsWithData<float16>(bnOp, fillTensorWithRandomData);
        bnOp->run();
        auto outputs = bnOp->getOutput(0);
        auto refOutputs = getReferenceOutput(bnOp, workspace());
        verifyOutputs<float16>(outputs, refOutputs);
    }
}

TEST_CASE_METHOD(SmaugTest, "SMV Post-FC Batch Norm", "[smvpool]") {
    auto bnOp = new SmvBatchNormOp("bn", workspace());

    SECTION("No tiling required") {
        TensorShape inputShape(
                { 1, 1024 }, DataLayout::NC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("input", inputShape);
        inputs->allocateStorage<float16>();
        workspace()->addTensor(inputs);
        bnOp->setInput(inputs, 0);
        createAndFillTensorsWithData<float16>(bnOp, fillTensorWithRandomData);
        bnOp->run();
        auto outputs = bnOp->getOutput(0);
        auto refOutputs = getReferenceOutput(bnOp, workspace());
        verifyOutputs<float16>(outputs, refOutputs);
    }

    SECTION("DimNC required") {
        TensorShape inputShape(
                { 1, 32768 }, DataLayout::NC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("input", inputShape);
        inputs->allocateStorage<float16>();
        workspace()->addTensor(inputs);
        bnOp->setInput(inputs, 0);
        createAndFillTensorsWithData<float16>(bnOp, fillTensorWithRandomData);
        bnOp->run();
        auto outputs = bnOp->getOutput(0);
        auto refOutputs = getReferenceOutput(bnOp, workspace());
        verifyOutputs<float16>(outputs, refOutputs);
    }
}
