#include "core/backend.h"
#include "operators/common.h"
#include "operators/relu_op.h"

#ifdef __cplusplus
extern "C" {
#endif

void ref_relu_f32(float* inputs, float* results, int input_size) {
    relu_loop:
    for (int i = 0; i < input_size; i++) {
        float value = inputs[i];
        if (value < 0.0) {
            results[i] = 0.0;
        } else {
            results[i] = value;
        }
    }
}

void ref_leaky_relu_f32(float* inputs, float* results, int input_size, float slope) {
    relu_loop:
    for (int i = 0; i < input_size; i++) {
        float value = inputs[i];
        if (value < 0.0) {
            results[i] = slope * value;
        } else {
            results[i] = value;
        }
    }
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
    if (slope == 1) {
        ref_relu_f32(inputs->data<float>(), outputs->data<float>(),
                     inputs->getShape().size());
    } else {
        ref_leaky_relu_f32(inputs->data<float>(), outputs->data<float>(),
                           inputs->getShape().size(), slope);
    }
}

}  // namespace smaug
