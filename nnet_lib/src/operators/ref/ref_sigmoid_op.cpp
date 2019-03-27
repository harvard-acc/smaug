#include "core/backend.h"
#include "operators/common.h"
#include "operators/sigmoid_op.h"
#include "operators/ref/ref_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// The logistic activation function.
//
// Operates on a single float.
ALWAYS_INLINE
float sigmoid_fxp(float a) {
    return 1.0 / (1.0 + exp(-a));
}

// The logistic activation function
void ref_sigmoid_f32(float* inputs, float* results, int input_size) {
    dmaLoad(inputs, inputs, input_size * sizeof(float));
    sigmoid_loop:
    for (int i = 0; i < input_size; i++) {
        results[i] = sigmoid_fxp(inputs[i]);
    }
    dmaStore(results, results, input_size * sizeof(float));
}

#ifdef __cplusplus
}
#endif

namespace smaug {

template <>
void SigmoidOp<ReferenceBackend>::run() {
    auto inputs = getInput(Inputs);
    auto outputs = getOutput(Outputs);
    assert(inputs->getShape() == outputs->getShape());
    float* inputData = inputs->data<float>();
    float* outputData = outputs->data<float>();
    MAP_ARRAY_TO_ACCEL(ref::kEltwiseOpHw, "inputs", inputData,
                       inputs->getShape().storageSize() * sizeof(float));
    MAP_ARRAY_TO_ACCEL(ref::kEltwiseOpHw, "results", outputData,
                       inputs->getShape().storageSize() * sizeof(float));
    INVOKE_KERNEL(kEltwiseOpHw, ref_sigmoid_f32, inputData, outputData,
                  inputs->getShape().size());
}

}  // namespace smaug
