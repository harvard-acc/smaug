#include "catch.hpp"
#include "core/backend.h"
#include "core/smaug_test.h"
#include "core/tensor.h"
#include "operators/smv/smv_eltwise_add_op.h"
#include "operators/smv/smv_test_common.h"

using namespace smaug;

namespace smaug {

class SmvEltwiseAddOpTest : public SmaugTest {
   public:
    using SmaugTest::SmaugTest;

    Tensor* getReferenceOutput(SmvEltwiseAddOp* eltAddOp) {
        auto inputs0 = eltAddOp->getInput(0);
        auto inputs1 = eltAddOp->getInput(1);
        auto inputs0Fp32 = convertFp16ToFp32Tensor(inputs0, workspace());
        auto inputs1Fp32 = convertFp16ToFp32Tensor(inputs1, workspace());

        // A reference operator is used to get the 'correct' output.
        auto refEltAddOp = new EltwiseAddOp<ReferenceBackend>(
                "ref_eltwise_add", workspace());
        refEltAddOp->setInput(inputs0Fp32, 0);
        refEltAddOp->setInput(inputs1Fp32, 1);
        refEltAddOp->createAllTensors();
        refEltAddOp->getOutput(0)->allocateStorage<float>();
        refEltAddOp->run();
        return convertFp32ToFp16Tensor(refEltAddOp->getOutput(0), workspace());
    }

    void doTest(std::vector<int> dims) {
        auto eltAddOp = new SmvEltwiseAddOp("eltwise_add", workspace());
        DataLayout layout = dims.size() == 4 ? NHWC : NC;
        TensorShape inputShape(dims, layout, SmvBackend::Alignment);
        Tensor* inputs0 = new Tensor("input0", inputShape);
        Tensor* inputs1 = new Tensor("input1", inputShape);
        inputs0->allocateStorage<float16>();
        inputs1->allocateStorage<float16>();
        workspace()->addTensor(inputs0);
        workspace()->addTensor(inputs1);
        eltAddOp->setInput(inputs0, 0);
        eltAddOp->setInput(inputs1, 1);
        createAndFillTensorsWithData<float16>(
                eltAddOp, fillTensorWithRandomData);
        eltAddOp->tile();
        eltAddOp->run();
        auto outputs = eltAddOp->getOutput(0);
        auto refOutputs = getReferenceOutput(eltAddOp);
        verifyOutputs<float16>(outputs, refOutputs);
    }
};

}  // namespace smaug

TEST_CASE_METHOD(SmvEltwiseAddOpTest,
                 "SMV Tiled 4D EltwiseAdd",
                 "[smveltadd]") {
    SECTION("No tiling required") { doTest({ 1, 8, 8, 8 }); }
    SECTION("DimNC tiling") { doTest({ 1, 32, 32, 32 }); }
    SECTION("DimNH tiling") { doTest({ 1, 64, 64, 32 }); }
    SECTION("DimNCH tiling") { doTest({ 1, 4, 2048, 16 }); }
}

TEST_CASE_METHOD(SmvEltwiseAddOpTest,
                 "SMV Tiled 2D EltwiseAdd",
                 "[smveltadd]") {
    SECTION("No tiling required") { doTest({ 1, 1024 }); }
    SECTION("DimNC tiling") { doTest({ 1, 32768 }); }
}

