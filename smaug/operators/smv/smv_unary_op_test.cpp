#include "catch.hpp"
#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/core/smaug_test.h"
#include "smaug/operators/smv/smv_test_common.h"
#include "smaug/operators/smv/smv_relu_op.h"
#include "smaug/operators/smv/smv_elu_op.h"
#include "smaug/operators/smv/smv_tanh_op.h"
#include "smaug/operators/smv/smv_sigmoid_op.h"
#include "smaug/operators/smv/smv_softmax_op.h"

using namespace smaug;

namespace smaug {

class SmvUnaryOpTest : public SmaugTest {
   public:
    using SmaugTest::SmaugTest;

    Tensor* getReferenceOutput(UnaryOp<SmvBackend>* unaryOp) {
        OpType opType = unaryOp->getOpType();
        auto input = unaryOp->getInput(0);
        auto input32 = convertFp16ToFp32Tensor(input, workspace());

        // A reference operator is used to get the 'correct' output.
        UnaryOp<ReferenceBackend>* refUnaryOp;
        if (opType == OpType::ReLU) {
            auto reluOp = dynamic_cast<SmvReluOp*>(unaryOp);
            float slope = reluOp->getSlope();
            ReluOp<ReferenceBackend>* refReluOp;
            if (slope != 0) {
                refReluOp =
                        new ReluOp<ReferenceBackend>("ref_lrelu", workspace());
                refReluOp->setSlope(slope);
            } else {
                refReluOp =
                        new ReluOp<ReferenceBackend>("ref_relu", workspace());
            }
            refUnaryOp = refReluOp;
        } else if (opType == OpType::ELU) {
            auto eluOp = dynamic_cast<SmvEluOp*>(unaryOp);
            auto refEluOp = new EluOp<ReferenceBackend>("ref_elu", workspace());
            refEluOp->setAlpha(eluOp->getAlpha());
            refUnaryOp = refEluOp;
        } else if (opType == OpType::SELU) {
            auto seluOp = dynamic_cast<SmvSeluOp*>(unaryOp);
            auto refSeluOp =
                    new SeluOp<ReferenceBackend>("ref_selu", workspace());
            refSeluOp->setAlpha(seluOp->getAlpha());
            refSeluOp->setLambda(seluOp->getLambda());
            refUnaryOp = refSeluOp;
        } else if (opType == OpType::Tanh) {
            refUnaryOp = new TanhOp<ReferenceBackend>("ref_tanh", workspace());
        } else if (opType == OpType::Sigmoid) {
            refUnaryOp =
                    new SigmoidOp<ReferenceBackend>("ref_sigmoid", workspace());
        } else if (opType == OpType::Softmax) {
            refUnaryOp =
                    new SoftmaxOp<ReferenceBackend>("ref_softmax", workspace());
        }
        refUnaryOp->setInput(input32, 0);
        refUnaryOp->createAllTensors();
        refUnaryOp->getOutput(0)->allocateStorage<float>();
        refUnaryOp->run();
        return convertFp32ToFp16Tensor(refUnaryOp->getOutput(0), workspace());
    }

    void doTest(OpType opType, std::vector<int> dims) {
        UnaryOp<SmvBackend>* unaryOp;
        if (opType == OpType::ReLU) {
            unaryOp = new SmvReluOp("relu", workspace());
        } else if (opType == LReLU) {
            auto reluOp = new SmvReluOp("lrelu", workspace());
            reluOp->setSlope(0.1);
            unaryOp = reluOp;
        } else if (opType == OpType::ELU) {
            auto eluOp = new SmvEluOp("elu", workspace());
            eluOp->setAlpha(0.1);
            unaryOp = eluOp;
        } else if (opType == OpType::SELU) {
            auto seluOp = new SmvSeluOp("selu", workspace());
            seluOp->setAlpha(1.6733);
            seluOp->setLambda(1.0507);
            unaryOp = seluOp;
        } else if (opType == OpType::Tanh) {
            unaryOp = new SmvSeluOp("tanh", workspace());
        } else if (opType == OpType::Sigmoid) {
            unaryOp = new SmvSigmoidOp("sigmoid", workspace());
        } else if (opType == OpType::Softmax) {
            unaryOp = new SmvSoftmaxOp("softmax", workspace());
        }
        DataLayout layout = dims.size() == 4 ? NHWC : NC;
        TensorShape inputShape(dims, layout, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("input", inputShape);
        inputs->allocateStorage<float16>();
        workspace()->addTensor(inputs);
        unaryOp->setInput(inputs, 0);
        createAndFillTensorsWithData<float16>(
                unaryOp, fillTensorWithRandomData);
        unaryOp->tile();
        unaryOp->run();
        auto outputs = unaryOp->getOutput(0);
        auto refOutputs = getReferenceOutput(unaryOp);
        verifyOutputs<float16>(outputs, refOutputs);
    }
};

}  // namespace smaug

TEST_CASE_METHOD(SmvUnaryOpTest, "SMV Tiled 4D Activations", "[smvunary]") {
    SECTION("No tiling required") {
        doTest(OpType::ReLU, { 1, 32, 32, 8 });
        doTest(OpType::LReLU, { 1, 32, 32, 8 });
        doTest(OpType::ELU, { 1, 32, 32, 8 });
        doTest(OpType::SELU, { 1, 32, 32, 8 });
        doTest(OpType::Tanh, { 1, 32, 32, 8 });
        doTest(OpType::Sigmoid, { 1, 32, 32, 8 });
    }
    SECTION("Tiling needed, last tile has the same shape") {
        doTest(OpType::ReLU, { 2, 16, 32, 32 });
        doTest(OpType::LReLU, { 2, 16, 32, 32 });
        doTest(OpType::ELU, { 2, 16, 32, 32 });
        doTest(OpType::SELU, { 2, 16, 32, 32 });
        doTest(OpType::Tanh, { 2, 16, 32, 32 });
        doTest(OpType::Sigmoid, { 2, 16, 32, 32 });
    }
    SECTION("Tiling needed, last tile has a different shape") {
        doTest(OpType::ReLU, { 2, 16, 32, 24 });
        doTest(OpType::LReLU, { 2, 16, 32, 24 });
        doTest(OpType::ELU, { 2, 16, 32, 24 });
        doTest(OpType::SELU, { 2, 16, 32, 24 });
        doTest(OpType::Tanh, { 2, 16, 32, 24 });
        doTest(OpType::Sigmoid, { 2, 16, 32, 24 });
    }
}

TEST_CASE_METHOD(SmvUnaryOpTest, "SMV Tiled 2D Activations", "[smvunary]") {
    SECTION("No tiling required") {
        doTest(OpType::ReLU, { 1, 1024 });
        doTest(OpType::LReLU, { 1, 1024 });
        doTest(OpType::ELU, { 1, 1024 });
        doTest(OpType::SELU, { 1, 1024 });
        doTest(OpType::Tanh, { 1, 1024 });
        doTest(OpType::Sigmoid, { 1, 1024 });
        doTest(OpType::Softmax, { 1, 1024 });
    }
    SECTION("Tiling needed, last tile has the same shape") {
        doTest(OpType::ReLU, { 2, 16384 });
        doTest(OpType::LReLU, { 2, 16384 });
        doTest(OpType::ELU, { 2, 16384 });
        doTest(OpType::SELU, { 2, 16384 });
        doTest(OpType::Tanh, { 2, 16384 });
        doTest(OpType::Sigmoid, { 2, 16384 });
        // The 8 inputs will be tiled into 4 tiles of 2 inputs each.
        doTest(OpType::Softmax, { 8, 8192 });
    }
    SECTION("Tiling needed, last tile has the same shape") {
        doTest(OpType::ReLU, { 2, 12288 });
        doTest(OpType::LReLU, { 2, 12288 });
        doTest(OpType::ELU, { 2, 12288 });
        doTest(OpType::SELU, { 2, 12288 });
        doTest(OpType::Tanh, { 2, 12288 });
        doTest(OpType::Sigmoid, { 2, 12288 });
        // The 9 inputs will be tiled into 3 tiles, which have 4, 4 and 1 inputs
        // respectively.
        doTest(OpType::Softmax, { 9, 4096 });
    }
}

