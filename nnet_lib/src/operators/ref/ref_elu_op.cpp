#include <cmath>

#include "core/backend.h"
#include "operators/common.h"
#include "operators/elu_op.h"
#include "operators/ref/ref_common.h"

#ifdef __cplusplus
extern "C" {
#endif

void ref_elu_f32(float* inputs, float* results, int input_size, float alpha) {
    dmaLoad(inputs, inputs, input_size * sizeof(float));
    elu_loop:
    for (int i = 0; i < input_size; i++) {
        float value = inputs[i];
        if (value < 0.0) {
            results[i] = alpha * (exp(value) - 1);
        } else {
            results[i] = value;
        }
    }
    dmaStore(results, results, input_size * sizeof(float));
}

void ref_selu_f32(float* inputs,
                  float* results,
                  int input_size,
                  float alpha,
                  float lambda) {
    dmaLoad(inputs, inputs, input_size * sizeof(float));
    ref_elu_f32(inputs, results, input_size, alpha);
    selu_loop:
    for (int i = 0; i < input_size; i++) {
        results[i] = lambda * results[i];
    }
    dmaStore(results, results, input_size * sizeof(float));
}

#ifdef __cplusplus
}
#endif

namespace smaug {

template <>
void EluOp<ReferenceBackend>::run() {
    auto inputs = getInput(Inputs);
    auto outputs = getOutput(Outputs);
    assert(inputs->getShape() == outputs->getShape());
    float* inputData = inputs->data<float>();
    float* outputData = outputs->data<float>();
    mapArrayToAccel(ref::kEltwiseOpHw, "inputs", inputData,
                    inputs->getShape().storageSize() * sizeof(float));
    mapArrayToAccel(ref::kEltwiseOpHw, "results", outputData,
                    inputs->getShape().storageSize() * sizeof(float));
    invokeKernel(ref::kEltwiseOpHw, ref_elu_f32, inputData, outputData,
                 inputs->getShape().size(), alpha);
}

template <>
void SeluOp<ReferenceBackend>::run() {
    auto inputs = getInput(Inputs);
    auto outputs = getOutput(Outputs);
    assert(inputs->getShape() == outputs->getShape());
    float* inputData = inputs->data<float>();
    float* outputData = outputs->data<float>();
    mapArrayToAccel(ref::kEltwiseOpHw, "inputs", inputData,
                    inputs->getShape().storageSize() * sizeof(float));
    mapArrayToAccel(ref::kEltwiseOpHw, "results", outputData,
                    inputs->getShape().storageSize() * sizeof(float));
    invokeKernel(ref::kEltwiseOpHw, ref_selu_f32, inputData, outputData,
                 inputs->getShape().size(), this->alpha, lambda);
}

}  // namespace smaug
