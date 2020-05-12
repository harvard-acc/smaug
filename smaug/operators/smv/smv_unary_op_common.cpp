#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/smv/smv_unary_op_common.h"
#include "smaug/operators/smv/smv_relu_op.h"
#include "smaug/operators/smv/smv_elu_op.h"
#include "smaug/operators/smv/smv_tanh_op.h"
#include "smaug/operators/smv/smv_sigmoid_op.h"
#include "smaug/operators/smv/smv_kernels.h"
#include "smaug/utility/debug_stream.h"

namespace smaug {
namespace smv {
namespace unary {

std::pair<activation_type, activation_param_t> getActivationParams(
        UnaryOp<SmvBackend>* op) {
    activation_type function;
    activation_param_t params;
    OpType opType = op->getOpType();
    if (opType == OpType::ReLU) {
        auto reluOp = dynamic_cast<SmvReluOp*>(op);
        params.slope = reluOp->getSlope();
        if (params.slope > 0)
            function = activation_type::LRELU;
        else
            function = activation_type::RELU;
    } else if (opType == OpType::ELU) {
        function = activation_type::ELU;
        auto eluOp = dynamic_cast<SmvEluOp*>(op);
        params.alpha = eluOp->getAlpha();
    } else if (opType == OpType::SELU) {
        function = activation_type::SELU;
        auto seluOp = dynamic_cast<SmvSeluOp*>(op);
        params.alpha = seluOp->getAlpha();
        params.lambda = seluOp->getLambda();
    } else if (opType == OpType::Tanh) {
        function = activation_type::TANH;
    } else if (opType == OpType::HardTanh) {
        function = activation_type::HARD_TANH;
        auto hardTanhOp = dynamic_cast<SmvHardTanhOp*>(op);
        params.min = hardTanhOp->getMin();
        params.max = hardTanhOp->getMax();
    } else if (opType == OpType::Sigmoid) {
        function = activation_type::SIGMOID;
    } else if (opType == OpType::Softmax) {
        assert(false && "Softmax should call its own run() implementation!");
    }
    return { function, params };
}

// The tile dispatcher for activation functions.
void runX(UnaryOp<SmvBackend>* op, TiledTensor& inputs, TiledTensor& outputs) {
    assert(inputs.size() == outputs.size());
    auto actParams = getActivationParams(op);
    setArrayMemTypeIfSimulating(
            smv::kEltwiseOpHw, "host_inputs", op->getInputsMemType());
    setArrayMemTypeIfSimulating(
            smv::kEltwiseOpHw, "host_results", op->getOutputsMemType());
    for (int i = 0; i < inputs.size(); i++) {
        dout(1) << "Input: " << i << ", output: " << i << "\n";
        Tensor* inputTile = inputs.getTileWithData(i);
        Tensor* outputTile = outputs[i];
        const TensorShape& inputShape = inputTile->getShape();
        const TensorShape& outputShape = outputTile->getShape();
        mapArrayToAccel(smv::kEltwiseOpHw, "host_inputs",
                        inputTile->data<float16>(),
                        inputShape.storageSize() * sizeof(float16));
        mapArrayToAccel(smv::kEltwiseOpHw, "host_results",
                        outputTile->data<float16>(),
                        outputShape.storageSize() * sizeof(float16));

        invokeKernel(smv::kEltwiseOpHw, smv_activation_fun_nc_vec_fxp,
                     inputTile->data<float16>(), outputTile->data<float16>(),
                     smv::spad0, smv::spad1, inputShape.storageSize(),
                     actParams.first, actParams.second);
    }
}

TiledTensor generateTiles(Tensor* tensor,
                          const TensorShape& tileShape,
                          Operator* op,
                          bool copyData) {
    const TensorShape& inputShape = tensor->getShape();
    int inputSize = inputShape.storageSize();
    int tileSize = tileShape.storageSize();
    int numTiles = std::ceil(inputSize * 1.0 / tileSize);
    TiledTensor tiledTensor(
            TensorShape({ 1, numTiles }, DataLayout::NC), tensor, true);
    int remainingSize = inputSize;
    int srcOffset = 0;
    for (auto tileIndex = tiledTensor.startIndex(); !tileIndex.end();
         ++tileIndex) {
        int currentTileSize = std::min(remainingSize, tileSize);
        TensorShape currentShape({ 1, currentTileSize },
                                 DataLayout::NC,
                                 tileShape.getAlignment());
        std::string tileName = op->getName() + ":" + tensor->getName() +
                               "/tile:" + std::to_string((int)tileIndex);
        Tensor* tile = new Tensor(tileName, currentShape);
        tile->allocateStorage(tensor->getDataType());
        tiledTensor.setTile(tileIndex, { srcOffset }, tile, copyData);
        srcOffset += currentTileSize;
        remainingSize -= currentTileSize;
    }
    op->getWorkspace()->addTiledTensor(tiledTensor);
    dout(1) << "  Tiled Tensor " << tensor->getName() << ":\n"
            << "    original tensor shape: " << tensor->getShape() << "\n"
            << "    tile shape " << tileShape
            << ", number of tiles: " << tiledTensor.size() << "\n";
    return tiledTensor;
}

std::array<TiledTensor, 2> doTiling(UnaryOp<SmvBackend>* op, bool copyData) {
    auto inputs = op->getInput(UnaryOp<SmvBackend>::Inputs);
    auto outputs = op->getOutput(UnaryOp<SmvBackend>::Outputs);
    // The tiling for unary operators can be greatly simplified in comparison to
    // other operators. The tile shape is determined as [1, spadSize].
    int maxTileSize =
            std::min(SmvBackend::SpadSize() / inputs->getDataTypeSize(),
                     inputs->getShape().storageSize());
    TensorShape tileShape(
            { 1, maxTileSize }, DataLayout::NC, SmvBackend::Alignment);
    TiledTensor tiledInputs = generateTiles(inputs, tileShape, op, copyData);
    TiledTensor tiledOutputs = generateTiles(outputs, tileShape, op, copyData);
    return { tiledInputs, tiledOutputs };
}

void run(UnaryOp<SmvBackend>* op, std::array<TiledTensor, 2>& tiledTensors) {
    auto inputs = op->getInput(UnaryOp<SmvBackend>::Inputs);
    auto outputs = op->getOutput(UnaryOp<SmvBackend>::Outputs);

    {
        auto stats = gem5::ScopedStats(
                stats::kTensorPrepStart, stats::kTensorPrepEnd);
        tiledTensors[0].copyDataToAllTiles();
    }

    runX(op, tiledTensors[0], tiledTensors[1]);

    {
        auto stats = gem5::ScopedStats(
                stats::kTensorFinalStart, stats::kTensorFinalEnd);
        flattenTiledTensor(tiledTensors[1], outputs);
    }
}

}  // namespace unary
}  // namespace smv
}  // namespace smaug

