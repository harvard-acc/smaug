#include "catch.hpp"
#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/core/smaug_test.h"
#include "smaug/operators/smv/smv_test_common.h"
#include "smaug/operators/smv/smv_pooling_op.h"
#include "smaug/operators/smv/smv_pooling_tiling.h"

using namespace smaug;

namespace smaug {

class SmvPoolingOpTest : public SmaugTest {
   public:
    using SmaugTest::SmaugTest;

    Tensor* getReferenceOutput(PoolingOp<SmvBackend>* poolOp) {
        OpType opType = poolOp->getOpType();
        auto input = poolOp->getInput(0);
        auto input32 = convertFp16ToFp32Tensor(input, workspace());

        // A reference pooling operator is used to get the 'correct' output.
        PoolingOp<ReferenceBackend>* refPoolOp;
        if (opType == MaxPooling)
            refPoolOp =
                    new MaxPoolingOp<ReferenceBackend>("ref_pool", workspace());
        else
            refPoolOp =
                    new AvgPoolingOp<ReferenceBackend>("ref_pool", workspace());
        refPoolOp->setPoolingSize(poolOp->getPoolingSize().first,
                                  poolOp->getPoolingSize().second);
        refPoolOp->setPoolingStride(poolOp->getPoolingStride().first,
                                    poolOp->getPoolingStride().second);
        refPoolOp->setInput(input32, 0);
        refPoolOp->createAllTensors();
        refPoolOp->getOutput(0)->allocateStorage<float>();
        refPoolOp->run();
        return convertFp32ToFp16Tensor(refPoolOp->getOutput(0), workspace());
    }

    void doTest(PoolingOp<SmvBackend>* poolOp, std::vector<int> inputDims) {
        TensorShape inputShape(inputDims, NHWC, SmvBackend::Alignment);
        Tensor* inputs = new Tensor("input", inputShape);
        inputs->allocateStorage<float16>();
        workspace()->addTensor(inputs);
        poolOp->setInput(inputs, 0);
        createAndFillTensorsWithData<float16>(poolOp, fillTensorWithRandomData);
        poolOp->tile();
        poolOp->run();
        auto outputs = poolOp->getOutput(0);
        auto refOutputs = getReferenceOutput(poolOp);
        verifyOutputs<float16>(outputs, refOutputs);
    }
};

}  // namespace smaug

TEST_CASE_METHOD(SmvPoolingOpTest, "SMV Tiled Pooling", "[smvpool]") {
    SECTION("Max pooling") {
        auto poolOp = new SmvMaxPoolingOp("pool", workspace());
        poolOp->setPoolingSize(2, 2);
        poolOp->setPoolingStride(2, 2);

        SECTION("No tiling required") { doTest(poolOp, { 1, 32, 32, 8 }); }
        SECTION("DimNC tiling on inputs, None for outputs") {
            // inputs tiled into 2 channelwise tiles.
            doTest(poolOp, { 1, 32, 32, 32 });
        }
        SECTION("DimNC tiling on both inputs and outputs") {
            // inputs/outputs tiled into 8 channelwise tiles.
            doTest(poolOp, { 1, 32, 32, 128 });
        }
        SECTION("DimNH tiling") {
            SECTION("Tiles have the same shape") {
                // Inputs/outputs tiled into 8 rowwise tiles.
                doTest(poolOp, { 1, 64, 64, 32 });
            }
            SECTION("Last tile has a different shape") {
                // Inputs/outputs tiled into 12 rowwise tiles.
                doTest(poolOp, { 1, 68, 68, 32 });
            }
        }
        SECTION("DimNW tiling") { doTest(poolOp, { 1, 512, 1024, 16 }); }
        SECTION("DimNHW tiling") { doTest(poolOp, { 1, 512, 512, 32 }); }
        SECTION("Channelwise tiling") {
            poolOp->setPoolingSize(16, 16);
            poolOp->setPoolingStride(16, 16);
            SECTION("DimNCH tiling") { doTest(poolOp, { 1, 64, 64, 128 }); }
            SECTION("DimNCW tiling") { doTest(poolOp, { 1, 64, 256, 128 }); }
        }
    }

    SECTION("Average pooling") {
        auto poolOp = new SmvAvgPoolingOp("pool", workspace());
        poolOp->setPoolingSize(2, 2);
        poolOp->setPoolingStride(2, 2);

        SECTION("No tiling required") { doTest(poolOp, { 1, 8, 8, 8 }); }
        SECTION("DimNC tiling on inputs, None for outputs") {
            // inputs tiled into 2 channelwise tiles.
            doTest(poolOp, { 1, 32, 32, 32 });
        }
        SECTION("DimNC tiling on both inputs and outputs") {
            // inputs/outputs tiled into 8 channelwise tiles.
            doTest(poolOp, { 1, 32, 32, 128 });
        }
        SECTION("DimNH tiling") {
            SECTION("Tiles have the same shape") {
                // Inputs/outputs tiled into 8 rowwise tiles.
                doTest(poolOp, { 1, 64, 64, 32 });
            }
            SECTION("Last tile has a different shape") {
                // Inputs/outputs tiled into 12 rowwise tiles.
                doTest(poolOp, { 1, 68, 68, 32 });
            }
        }
        SECTION("DimNW tiling") { doTest(poolOp, { 1, 512, 1024, 16 }); }
        SECTION("DimNHW tiling") { doTest(poolOp, { 1, 512, 512, 32 }); }
        SECTION("Channelwise tiling") {
            poolOp->setPoolingSize(16, 16);
            poolOp->setPoolingStride(16, 16);
            SECTION("DimNCH tiling") { doTest(poolOp, { 1, 64, 64, 128 }); }
            SECTION("DimNCW tiling") { doTest(poolOp, { 1, 64, 256, 128 }); }
        }
    }
}

