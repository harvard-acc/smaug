#include "core/backend.h"
#include "operators/common.h"
#include "operators/relu_op.h"

#ifdef __cplusplus
extern "C" {
#endif

void ref_relu(float* inputs, float* results, int input_size) {
    dmaLoad(inputs, inputs, input_size * sizeof(float));
    relu_loop:
    for (int i = 0; i < input_size; i++) {
        float value = inputs[i];
        if (value < 0.0) {
            results[i] = 0.0;
        } else {
            results[i] = value;
        }
    }
    dmaStore(results, results, input_size * sizeof(float));
}

void ref_leaky_relu(float* inputs,
                    float* results,
                    int input_size,
                    float slope) {
    dmaLoad(inputs, inputs, input_size * sizeof(float));
    relu_loop:
    for (int i = 0; i < input_size; i++) {
        float value = inputs[i];
        if (value < 0.0) {
            results[i] = slope * value;
        } else {
            results[i] = value;
        }
    }
    dmaStore(results, results, input_size * sizeof(float));
}

#ifdef __cplusplus
}
#endif

namespace smaug {

template <>
void ReluOp<ReferenceBackend>::run() {
    auto inputs = getInput(Inputs);
    auto outputs = getOutput(Outputs);
    assert(inputs->getShape() == outputs->getShape());
    float* inputData = inputs->data<float>();
    float* outputData = outputs->data<float>();
    mapArrayToAccel(ref::kEltwiseOpHw, "inputs", inputData,
                    inputs->getShape().storageSize() * sizeof(float));
    mapArrayToAccel(ref::kEltwiseOpHw, "results", outputData,
                    inputs->getShape().storageSize() * sizeof(float));
    if (slope == 0) {
        invokeKernel(ref::kEltwiseOpHw, ref_relu, inputData, outputData,
                     inputs->getShape().size());
    } else {
        invokeKernel(ref::kEltwiseOpHw, ref_leaky_relu, inputData, outputData,
                     inputs->getShape().size(), slope);
    }
}

}  // namespace smaug
