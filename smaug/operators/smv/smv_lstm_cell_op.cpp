#include "smaug/operators/smv/smv_lstm_op.h"

namespace smaug {

void SmvLSTMCellOp::tile() {}

void SmvLSTMCellOp::run() {
    auto inputsX = getInput(InputsX);
    auto inputsH = getInput(InputsH);
    auto inputsC = getInput(InputsC);
    auto weightsX = getInput(WeightsX);
    auto weightsH = getInput(WeightsH);
    auto outputsH = getOutput(OutputsH);
    auto outputsC = getOutput(OutputsC);

    // Implement the single step LSTM operation.
    float16* inputsXDataFp16 = inputsX->data<float16>();
    float16* inputsHDataFp16 = inputsH->data<float16>();
    float16* outputsHDataFp16 = outputsH->data<float16>();
    float16* outputsCDataFp16 = outputsC->data<float16>();
    float* inputsXData = inputsX->data<float16>();
    float* inputsHData = inputsH->data<float16>();
    float* outputsHData = outputsH->data<float16>();
    float* outputsCData = outputsC->data<float16>();
    // Convert FP16 to FP32.
    host_load_fp16(inputsXData, inputsXDataFp16, inputsX->getTotalDim(), 0, 0);
    host_load_fp16(inputsHData, inputsHDataFp16, inputsH->getTotalDim(), 0, 0);
    // Equation.
    for (int b = 0; b < batches; b++) {
        for (int o = 0; o < outputs; o++) {
            // Matrix vector.
            for (int i = 0; i < numInputs; i++) {
              // Wx * X.
            }
            for (int i = 0; i < numHiddens; i++) {
              // Wh * H.
            }
        }
        // Activation functions.
    }
    // Convert results from FP32 to FP16.
    host_store_fp16(outputsHData, outputsHDataFp16, results_size, 0, 0);
    host_store_fp16(outputsCData, outputsCDataFp16, results_size, 0, 0);
}

}  // namespace smaug
