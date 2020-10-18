#include "smaug/operators/smv/smv_less_op.h"
#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/smv/smv_kernels.h"
#include "smaug/operators/smv/smv_unary_op_common.h"
#include "smaug/utility/debug_stream.h"

namespace smaug {

void SmvLessOp::runX(TiledTensor& inputs0,
                     TiledTensor& inputs1,
                     TiledTensor& outputs) {
    assert(inputs0.size() == inputs1.size() &&
           inputs0.size() == outputs.size());
    setArrayMemTypeIfSimulating(
            smv::kEltwiseOpHw, "host_inputs0", getInputsMemType());
    setArrayMemTypeIfSimulating(
            smv::kEltwiseOpHw, "host_inputs1", getInputsMemType());
    setArrayMemTypeIfSimulating(
            smv::kEltwiseOpHw, "host_results", getOutputsMemType());
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
                        outputTile->data<bool>(),
                        outputShape.storageSize() * sizeof(bool));

        invokeKernel(smv::kEltwiseOpHw, smv_less_nc_vec_fxp,
                     input0Tile->data<float16>(), input1Tile->data<float16>(),
                     outputTile->data<bool>(), smv::spad0, smv::spad1,
                     reinterpret_cast<bool*>(smv::spad2),
                     inputShape.storageSize());
    }
}

void SmvLessOp::tile() {
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
    tiledTensors[0] =
            generateTiledTensorPerBatchNC(inputs0, tileShape, this, false);
    tiledTensors[1] =
            generateTiledTensorPerBatchNC(inputs1, tileShape, this, false);
    tiledTensors[2] =
            generateTiledTensorPerBatchNC(outputs, tileShape, this, false);
}

void SmvLessOp::run() {
    auto inputs0 = getInput(Input0);
    auto inputs1 = getInput(Input1);
    auto outputs = getOutput(Outputs);
    const TensorShape& inputs0Shape = inputs0->getShape();
    const TensorShape& inputs1Shape = inputs1->getShape();
    const TensorShape& outputsShape = outputs->getShape();
    assert(inputs0Shape == inputs1Shape && inputs0Shape == outputsShape);

    {
        auto stats = gem5::ScopedStats(
                stats::kTensorPrepStart, stats::kTensorPrepEnd);
        tiledTensors[0].copyDataToAllTiles();
        tiledTensors[1].copyDataToAllTiles();
    }

    runX(tiledTensors[0], tiledTensors[1], tiledTensors[2]);

    {
        auto stats = gem5::ScopedStats(
                stats::kTensorFinalStart, stats::kTensorFinalEnd);
        flattenTiledTensor(tiledTensors[2], outputs);
    }
}

void SmvLessEqualOp::runX(TiledTensor& inputs0,
                          TiledTensor& inputs1,
                          TiledTensor& outputs) {
    assert(inputs0.size() == inputs1.size() &&
           inputs0.size() == outputs.size());
    setArrayMemTypeIfSimulating(
            smv::kEltwiseOpHw, "host_inputs0", getInputsMemType());
    setArrayMemTypeIfSimulating(
            smv::kEltwiseOpHw, "host_inputs1", getInputsMemType());
    setArrayMemTypeIfSimulating(
            smv::kEltwiseOpHw, "host_results", getOutputsMemType());
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
                        outputTile->data<bool>(),
                        outputShape.storageSize() * sizeof(bool));

        invokeKernel(smv::kEltwiseOpHw, smv_less_equal_nc_vec_fxp,
                     input0Tile->data<float16>(), input1Tile->data<float16>(),
                     outputTile->data<bool>(), smv::spad0, smv::spad1,
                     reinterpret_cast<bool*>(smv::spad2),
                     inputShape.storageSize());
    }
}

void SmvLessEqualOp::tile() {
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
    tiledTensors[0] =
            generateTiledTensorPerBatchNC(inputs0, tileShape, this, false);
    tiledTensors[1] =
            generateTiledTensorPerBatchNC(inputs1, tileShape, this, false);
    tiledTensors[2] =
            generateTiledTensorPerBatchNC(outputs, tileShape, this, false);
}

void SmvLessEqualOp::run() {
    auto inputs0 = getInput(Input0);
    auto inputs1 = getInput(Input1);
    auto outputs = getOutput(Outputs);
    const TensorShape& inputs0Shape = inputs0->getShape();
    const TensorShape& inputs1Shape = inputs1->getShape();
    const TensorShape& outputsShape = outputs->getShape();
    assert(inputs0Shape == inputs1Shape && inputs0Shape == outputsShape);

    {
        auto stats = gem5::ScopedStats(
                stats::kTensorPrepStart, stats::kTensorPrepEnd);
        tiledTensors[0].copyDataToAllTiles();
        tiledTensors[1].copyDataToAllTiles();
    }

    runX(tiledTensors[0], tiledTensors[1], tiledTensors[2]);

    {
        auto stats = gem5::ScopedStats(
                stats::kTensorFinalStart, stats::kTensorFinalEnd);
        flattenTiledTensor(tiledTensors[2], outputs);
    }
}

}  // namespace smaug
