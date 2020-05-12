#include "catch.hpp"
#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/core/smaug_test.h"
#include "smaug/operators/smv/smv_test_common.h"
#include "smaug/operators/smv/smv_inner_product_op.h"
#include "smaug/operators/smv/smv_inner_product_tiling.h"

using namespace smaug;

namespace smaug {

class SmvInnerProductOpTest : public SmaugTest {
   public:
    using SmaugTest::SmaugTest;

    Tensor* getReferenceOutput(SmvInnerProductOp* fcOp) {
        auto input = fcOp->getInput(0);
        auto weights = fcOp->getInput(1);
        auto input32 = convertFp16ToFp32Tensor(input, workspace());
        auto weights32 = convertFp16ToFp32Tensor(weights, workspace());

        // A reference inner product operator is used to get the 'correct'
        // output.
        auto refFcOp =
                new InnerProductOp<ReferenceBackend>("ref_fc", workspace());
        refFcOp->setActivation(fcOp->getActivation());
        refFcOp->setInput(input32, 0);
        refFcOp->setInput(weights32, 1);
        refFcOp->setNumOutputs(fcOp->getNumOutputs());
        refFcOp->createAllTensors();
        refFcOp->getOutput(0)->allocateStorage<float>();
        refFcOp->run();

        auto refOutput = refFcOp->getOutput(0);
        return convertFp32ToFp16Tensor(refOutput, workspace());
    }

    void doTest(std::vector<int> inputDims, int numNeurons) {
        auto fcOp = new SmvInnerProductOp("fc", workspace());
        TensorShape inputShape(
                inputDims, DataLayout::NC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("input", inputShape);
        workspace()->addTensor(inputs);
        fcOp->setInput(inputs, 0);
        fcOp->setNumOutputs(numNeurons);
        inputs->allocateStorage<float16>();
        createAndFillTensorsWithData<float16>(fcOp, fillTensorWithRandomData);
        fcOp->tile();
        fcOp->run();
        auto outputs = fcOp->getOutput(0);
        auto refOutputs = getReferenceOutput(fcOp);
        verifyOutputs<float16>(outputs, refOutputs);
    }

    void doFusionTest(
            std::vector<int> inputDims,
            int numNeurons,
            ActivationInfo actInfo = ActivationInfo(activation_type::ELU)) {
        auto fcOp = new SmvInnerProductOp("fc", workspace());
        fcOp->setActivation(actInfo);
        TensorShape inputShape(
                inputDims, DataLayout::NC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("input", inputShape);
        workspace()->addTensor(inputs);
        fcOp->setInput(inputs, 0);
        fcOp->setNumOutputs(numNeurons);
        inputs->allocateStorage<float16>();
        createAndFillTensorsWithData<float16>(fcOp, fillTensorWithRandomData);
        fcOp->tile();
        fcOp->run();
        auto outputs = fcOp->getOutput(0);
        auto refOutputs = getReferenceOutput(fcOp);
        verifyOutputs<float16>(outputs, refOutputs);
    }
};

}  // namespace smaug

TEST_CASE_METHOD(SmvInnerProductOpTest, "SMV tiled inner product", "[smvfc]") {
    SECTION("No tiling required") {
        doTest({1, 256}, 32);
    }

    SECTION("DimN tiling for weights, None for inputs") {
        // Weights are tiled into 2 tiles.
        doTest({1, 256}, 128);
    }

    SECTION("DimNC tiling for weights, None for inputs") {
        // Weights are tiled into 16 neuron-wise tiles and 2 activation-wise
        // tiles.
        doTest({ 1, 4096 }, 128);
    }

    SECTION("DimNC tiling for weights and inputs") {
        // Inputs/weights are tiled into 16 activation-wise tiles, weights are
        // also tiled into 32 neuron-wise tiles.
        doTest({ 1, 32768 }, 256);
    }
}

TEST_CASE_METHOD(SmvInnerProductOpTest,
                 "SMV tiled inner product with fused activation",
                 "[smvfc]") {
    SECTION("No tiling required") { doFusionTest({ 1, 256 }, 32); }

    SECTION("DimN tiling for weights, None for inputs") {
        doFusionTest({ 1, 256 }, 128);
    }

    SECTION("DimNC tiling for weights, None for inputs") {
        doFusionTest({ 1, 4096 }, 128);
    }

    SECTION("DimNC tiling for weights and inputs") {
        doFusionTest({ 1, 32768 }, 256);
    }
}
