#include "core/backend.h"
#include "operators/common.h"
#include "operators/smv/smv_eltwise_add_op.h"
#include "operators/smv/smv_unary_op_common.h"
#include "operators/smv/smv_kernels.h"
#include "utility/debug_stream.h"

namespace smaug {

// The tile dispatcher for elementwise addition.
void SmvEltwiseAddOp::runX(TiledTensor& inputs0,
                           TiledTensor& inputs1,
                           TiledTensor& outputs) {
    assert(inputs0.size() == inputs1.size() &&
           inputs0.size() == outputs.size());
    setArrayMemoryType(smv::kEltwiseOpHw, "host_inputs0", getInputsMemType());
    setArrayMemoryType(smv::kEltwiseOpHw, "host_inputs1", getInputsMemType());
    setArrayMemoryType(smv::kEltwiseOpHw, "host_results", getOutputsMemType());
    for (int i = 0; i < inputs0.size(); i++) {
        dout(1) << "Input0: " << i << ", input1: " << i << ", output: " << i
                << "\n";
        Tensor* input0Tile = inputs0.getTileWithData(i);
        Tensor* input1Tile = inputs1.getTileWithData(i);
        Tensor* outputTile = outputs[i];
        const TensorShape& inputShape = input0Tile->getShape();
        const TensorShape& outputShape = outputTile->getShape();
        mapArrayToAccel(smv::kEltwiseOpHw, "host_inputs0",
                        input0Tile->data<float16>(),
                        inputShape.storageSize() * sizeof(float16));
        mapArrayToAccel(smv::kEltwiseOpHw, "host_inputs1",
                        input1Tile->data<float16>(),
                        inputShape.storageSize() * sizeof(float16));
        mapArrayToAccel(smv::kEltwiseOpHw, "host_results",
                        outputTile->data<float16>(),
                        outputShape.storageSize() * sizeof(float16));

        invokeKernel(smv::kEltwiseOpHw, smv_eltwise_add_nc_vec_fxp,
                     input0Tile->data<float16>(), input1Tile->data<float16>(),
                     outputTile->data<float16>(), smv::spad0, smv::spad1,
                     smv::spad2, inputShape.storageSize());
    }
}

void SmvEltwiseAddOp::tile() {
    // We reuse the unary op tiler for the elementwise addition operator.
    using namespace smaug::smv::unary;
    auto inputs0 = getInput(Input0);
    auto inputs1 = getInput(Input1);
    auto outputs = getOutput(Outputs);
    int maxTileSize =
            std::min(SmvBackend::SpadSize() / inputs0->getDataTypeSize(),
                     inputs0->getShape().storageSize());
    TensorShape tileShape(
            { 1, maxTileSize }, DataLayout::NC, SmvBackend::Alignment);
    tiledTensors[0] = generateTiles(inputs0, tileShape, this);
    tiledTensors[1] = generateTiles(inputs1, tileShape, this);
    tiledTensors[2] = generateTiles(outputs, tileShape, this);
}

void SmvEltwiseAddOp::run() {
    auto inputs0 = getInput(Input0);
    auto inputs1 = getInput(Input1);
    auto outputs = getOutput(Outputs);
    const TensorShape& inputs0Shape = inputs0->getShape();
    const TensorShape& inputs1Shape = inputs1->getShape();
    const TensorShape& outputsShape = outputs->getShape();
    assert(inputs0Shape == inputs1Shape && inputs0Shape == outputsShape);

    runX(tiledTensors[0], tiledTensors[1], tiledTensors[2]);
    flattenTiledTensor(tiledTensors[2], outputs);
}

}  // namespace smaug
