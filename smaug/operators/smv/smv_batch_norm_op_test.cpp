#include "catch.hpp"
#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/core/smaug_test.h"
#include "smaug/operators/smv/smv_test_common.h"
#include "smaug/operators/smv/smv_batch_norm_op.h"

using namespace smaug;

namespace smaug {

class SmvBatchNormOpTest : public SmaugTest {
   public:
    using SmaugTest::SmaugTest;

    Tensor* getReferenceOutput(SmvBatchNormOp* bnOp) {
        auto input16 = bnOp->getInput(0);
        auto mean16 = bnOp->getInput(SmvBatchNormOp::Mean);
        auto variance16 = bnOp->getInput(SmvBatchNormOp::Variance);
        auto gamma16 = bnOp->getInput(SmvBatchNormOp::Gamma);
        auto beta16 = bnOp->getInput(SmvBatchNormOp::Beta);
        auto input32 = convertFp16ToFp32Tensor(input16, workspace());
        auto mean32 = convertFp16ToFp32Tensor(mean16, workspace());
        auto variance32 = convertFp16ToFp32Tensor(variance16, workspace());
        auto gamma32 = convertFp16ToFp32Tensor(gamma16, workspace());
        auto beta32 = convertFp16ToFp32Tensor(beta16, workspace());

        // A reference batch norm operator is used to get the 'correct' output.
        BatchNormOp<ReferenceBackend>* refBnOp =
                new BatchNormOp<ReferenceBackend>("ref_bn", workspace());
        refBnOp->setActivation(bnOp->getActivation());
        refBnOp->setInput(input32, 0);
        refBnOp->setInput(mean32, SmvBatchNormOp::Mean);
        refBnOp->setInput(variance32, SmvBatchNormOp::Variance);
        refBnOp->setInput(gamma32, SmvBatchNormOp::Gamma);
        refBnOp->setInput(beta32, SmvBatchNormOp::Beta);
        refBnOp->createAllTensors();
        refBnOp->getOutput(0)->allocateStorage<float>();
        refBnOp->run();
        return convertFp32ToFp16Tensor(refBnOp->getOutput(0), workspace());
    }

    void doTest(std::vector<int> dims) {
        auto bnOp = new SmvBatchNormOp("bn", workspace());
        DataLayout layout = dims.size() == 4 ? NHWC : NC;
        TensorShape inputShape(dims, layout, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("input", inputShape);
        inputs->allocateStorage<float16>();
        workspace()->addTensor(inputs);
        bnOp->setInput(inputs, 0);
        createAndFillTensorsWithData<float16>(bnOp, fillTensorWithRandomData);
        bnOp->tile();
        bnOp->run();
        auto outputs = bnOp->getOutput(0);
        auto refOutputs = getReferenceOutput(bnOp);
        verifyOutputs<float16>(outputs, refOutputs);
    }

    void doFusionTest(std::vector<int> dims) {
        auto bnOp = new SmvBatchNormOp("bn", workspace());
        ActivationInfo actInfo;
        actInfo.function = activation_type::ELU;
        bnOp->setActivation(actInfo);
        DataLayout layout = dims.size() == 4 ? NHWC : NC;
        TensorShape inputShape(dims, layout, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("input", inputShape);
        inputs->allocateStorage<float16>();
        workspace()->addTensor(inputs);
        bnOp->setInput(inputs, 0);
        createAndFillTensorsWithData<float16>(bnOp, fillTensorWithRandomData);
        bnOp->tile();
        bnOp->run();
        auto outputs = bnOp->getOutput(0);
        auto refOutputs = getReferenceOutput(bnOp);
        verifyOutputs<float16>(outputs, refOutputs);
    }
};

}  // namespace smaug

TEST_CASE_METHOD(SmvBatchNormOpTest,
                 "SMV Post-Conv Batch Norm",
                 "[smvpool]") {
    SECTION("No tiling required") { doTest({ 1, 32, 32, 16 }); }
    SECTION("DimNC tiling") { doTest({ 1, 16, 16, 128 }); }
    SECTION("DimNH tiling") { doTest({ 1, 64, 64, 32 }); }
    SECTION("DimNW tiling") { doTest({ 1, 64, 1024, 32 }); }
    SECTION("DimNHW tiling") { doTest({ 1, 128, 128, 64 }); }
    SECTION("DimNCH tiling") { doTest({ 1, 64, 64, 512 }); }
    SECTION("DimNCW tiling") { doTest({ 1, 64, 512, 512 }); }
}

TEST_CASE_METHOD(SmvBatchNormOpTest, "SMV Post-FC Batch Norm", "[smvpool]") {
    SECTION("No tiling required") { doTest({ 1, 1024 }); }
    SECTION("DimNC required") { doTest({ 1, 32768 }); }
}

TEST_CASE_METHOD(SmvBatchNormOpTest,
                 "SMV Post-Conv Batch Norm with fused activation",
                 "[smvpool]") {
    SECTION("No tiling required") { doFusionTest({ 1, 32, 32, 16 }); }
    SECTION("DimNC tiling") { doFusionTest({ 1, 16, 16, 128 }); }
    SECTION("DimNH tiling") { doFusionTest({ 1, 64, 64, 32 }); }
    SECTION("DimNW tiling") { doFusionTest({ 1, 64, 1024, 32 }); }
    SECTION("DimNHW tiling") { doFusionTest({ 1, 128, 128, 64 }); }
    SECTION("DimNCH tiling") { doFusionTest({ 1, 64, 64, 512 }); }
    SECTION("DimNCW tiling") { doFusionTest({ 1, 64, 512, 512 }); }
}

TEST_CASE_METHOD(SmvBatchNormOpTest,
                 "SMV Post-FC Batch Norm with fused activation",
                 "[smvpool]") {
    SECTION("No tiling required") { doFusionTest({ 1, 1024 }); }
    SECTION("DimNC required") { doFusionTest({ 1, 32768 }); }
}
