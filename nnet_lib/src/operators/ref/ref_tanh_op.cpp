#include "core/backend.h"
#include "operators/common.h"
#include "operators/tanh_op.h"
#include "operators/ref/ref_common.h"

#ifdef __cplusplus
extern "C" {
#endif

void ref_tanh_f32(float* inputs, float* results, int input_size) {
    int i;
    dmaLoad(inputs, inputs, input_size * sizeof(float));
    tanh_act_loop1:
    for (i = 0; i < input_size; i++) {
        results[i] = 2 * inputs[i];
    }

    ref_sigmoid_f32(results, results, input_size);

    tanh_act_loop2:
    for (i = 0; i < input_size; i++) {
        results[i] = 2 * results[i] - 1;
    }
    dmaStore(results, results, input_size * sizeof(float));
}

void ref_hard_tanh_f32(
        float* inputs, float* results, int input_size, float min, float max) {
    dmaLoad(inputs, inputs, input_size * sizeof(float));
    hard_tanh_loop:
    for (int i = 0; i < input_size; i++) {
        float value = inputs[i];
        results[i] = (value < min) ? min : (value > max) ? max : value;
    }
    dmaStore(results, results, input_size * sizeof(float));
}

#ifdef __cplusplus
}
#endif

namespace smaug {

template <>
void TanhOp<ReferenceBackend>::run() {
    auto inputs = getInput(Inputs);
    auto outputs = getOutput(Outputs);
    assert(inputs->getShape() == outputs->getShape());
    float* inputData = inputs->data<float>();
    float* outputData = outputs->data<float>();
    MAP_ARRAY_TO_ACCEL(ref::kEltwiseOpHw, "inputs", inputData,
                       inputs->getShape().storageSize() * sizeof(float));
    MAP_ARRAY_TO_ACCEL(ref::kEltwiseOpHw, "results", outputData,
                       inputs->getShape().storageSize() * sizeof(float));
    INVOKE_KERNEL(kEltwiseOpHw, ref_tanh_f32, inputData, outputData,
                  inputs->getShape().size());
}

template <>
void HardTanhOp<ReferenceBackend>::run() {
    auto inputs = getInput(Inputs);
    auto outputs = getOutput(Outputs);
    assert(inputs->getShape() == outputs->getShape());
    float* inputData = inputs->data<float>();
    float* outputData = outputs->data<float>();
    MAP_ARRAY_TO_ACCEL(ref::kEltwiseOpHw, "inputs", inputData,
                       inputs->getShape().storageSize() * sizeof(float));
    MAP_ARRAY_TO_ACCEL(ref::kEltwiseOpHw, "results", outputData,
                       inputs->getShape().storageSize() * sizeof(float));
    INVOKE_KERNEL(kEltwiseOpHw, ref_hard_tanh_f32, inputData, outputData,
                  inputs->getShape().size(), min, max);
}

}  // namespace smaug
