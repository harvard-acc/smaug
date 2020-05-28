#include "catch.hpp"
#include "smaug/core/backend.h"
#include "smaug/core/smaug_test.h"
#include "smaug/core/tensor.h"
#include "smaug/operators/smv/smv_eltwise_add_op.h"
#include "smaug/operators/smv/smv_eltwise_mul_op.h"
#include "smaug/operators/smv/smv_less_op.h"
#include "smaug/operators/smv/smv_greater_op.h"
#include "smaug/operators/smv/smv_test_common.h"

using namespace smaug;

namespace smaug {

class SmvEltwiseOpsTest : public SmaugTest {
   public:
    using SmaugTest::SmaugTest;

    // A reference operator is used to get the 'correct' output.
    Tensor* getReferenceOutput(Operator* eltOp, OpType opType) {
        auto inputs0 = eltOp->getInput(0);
        auto inputs1 = eltOp->getInput(1);
        auto inputs0Fp32 = convertFp16ToFp32Tensor(inputs0, workspace());
        auto inputs1Fp32 = convertFp16ToFp32Tensor(inputs1, workspace());

        Operator* refEltOp;
        bool boolOutput = true;
        switch(opType) {
          case EltwiseAdd:
              refEltOp = new EltwiseAddOp<ReferenceBackend>(
                      "ref_eltwise_add", workspace());
              boolOutput = false;
              break;
          case EltwiseMul:
              refEltOp = new EltwiseMulOp<ReferenceBackend>(
                      "ref_eltwise_mul", workspace());
              boolOutput = false;
              break;
          case Less:
              refEltOp = new LessOp<ReferenceBackend>("ref_less", workspace());
              break;
          case LessEqual:
              refEltOp = new LessEqualOp<ReferenceBackend>(
                      "ref_less_equal", workspace());
              break;
          case Greater:
              refEltOp = new GreaterOp<ReferenceBackend>(
                      "ref_greater", workspace());
              break;
          case GreaterEqual:
              refEltOp = new GreaterEqualOp<ReferenceBackend>(
                      "ref_greater_equal", workspace());
              break;
          default:
              assert(false && "Unexpected OpType!");
        }
        refEltOp->setInput(inputs0Fp32, 0);
        refEltOp->setInput(inputs1Fp32, 1);
        refEltOp->createAllTensors();
        if (boolOutput)
            refEltOp->getOutput(0)->allocateStorage<bool>();
        else
            refEltOp->getOutput(0)->allocateStorage<float>();
        refEltOp->run();
        if (boolOutput)
            return refEltOp->getOutput(0);
        else
            return convertFp32ToFp16Tensor(refEltOp->getOutput(0), workspace());
    }

    void doSingleTest(const std::vector<int>& dims, OpType opType) {
        Operator* eltOp;
        bool boolOutput = true;
        switch (opType) {
            case EltwiseAdd:
                eltOp = new SmvEltwiseAddOp("eltwise_add", workspace());
                boolOutput = false;
                break;
            case EltwiseMul:
                eltOp = new SmvEltwiseMulOp("eltwise_mul", workspace());
                boolOutput = false;
                break;
            case Less:
                eltOp = new SmvLessOp("less", workspace());
                break;
            case LessEqual:
                eltOp = new SmvLessEqualOp("less_equal", workspace());
                break;
            case Greater:
                eltOp = new SmvGreaterOp("greater", workspace());
                break;
            case GreaterEqual:
                eltOp = new SmvGreaterEqualOp("greater_equal", workspace());
                break;
            default:
                assert(false && "Unexpected OpType!");
        }
        DataLayout layout = dims.size() == 4 ? NHWC : NC;
        TensorShape inputShape(dims, layout, SmvBackend::Alignment);
        Tensor* inputs0 = new Tensor("input0", inputShape);
        Tensor* inputs1 = new Tensor("input1", inputShape);
        inputs0->allocateStorage<float16>();
        inputs1->allocateStorage<float16>();
        workspace()->addTensor(inputs0);
        workspace()->addTensor(inputs1);
        eltOp->setInput(inputs0, 0);
        eltOp->setInput(inputs1, 1);
        eltOp->createAllTensors();
        if (boolOutput)
            eltOp->getOutput(0)->allocateStorage<bool>();
        else
            eltOp->getOutput(0)->allocateStorage<float16>();
        for (auto input : eltOp->getInputs()) {
            Tensor* tensor = dynamic_cast<Tensor*>(input);
            fillTensorWithRandomData(tensor);
        }
        eltOp->tile();
        eltOp->run();
        auto outputs = eltOp->getOutput(0);
        auto refOutputs = getReferenceOutput(eltOp, opType);
        if (boolOutput)
            verifyOutputs<bool>(outputs, refOutputs);
        else
            verifyOutputs<float16>(outputs, refOutputs);
    }

    void doTest(const std::vector<int>& dims) {
        doSingleTest(dims, EltwiseAdd);
        doSingleTest(dims, EltwiseMul);
        doSingleTest(dims, Less);
        doSingleTest(dims, LessEqual);
        doSingleTest(dims, Greater);
        doSingleTest(dims, GreaterEqual);
    }
};

}  // namespace smaug

TEST_CASE_METHOD(SmvEltwiseOpsTest, "SMV Tiled 4D Eltwise Ops", "[smveltops]") {
    SECTION("No tiling required") { doTest({ 1, 8, 8, 8 }); }
    SECTION("DimNC tiling") { doTest({ 1, 32, 32, 32 }); }
    SECTION("DimNH tiling") { doTest({ 1, 64, 64, 32 }); }
    SECTION("DimNCH tiling") { doTest({ 1, 4, 2048, 16 }); }
}

TEST_CASE_METHOD(SmvEltwiseOpsTest, "SMV Tiled 2D Eltwise Ops", "[smveltops]") {
    SECTION("No tiling required") { doTest({ 1, 1024 }); }
    SECTION("DimNC tiling") { doTest({ 1, 32768 }); }
}

