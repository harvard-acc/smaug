#include <cmath>

#include "core/backend.h"
#include "operators/common.h"
#include "operators/elu_op.h"
#include "operators/ref/ref_common.h"

#ifdef __cplusplus
extern "C" {
#endif

void ref_elu_f32(float* inputs, float* results, int input_size, float alpha) {
    elu_loop:
    for (int i = 0; i < input_size; i++) {
        float value = inputs[i];
        if (value < 0.0) {
            results[i] = alpha * (exp(value) - 1);
        } else {
            results[i] = value;
        }
    }
}

void ref_selu_f32(float* inputs,
                  float* results,
                  int input_size,
                  float alpha,
                  float lambda) {
    ref_elu_f32(inputs, results, input_size, alpha);
    selu_loop:
    for (int i = 0; i < input_size; i++) {
        results[i] = lambda * results[i];
    }
}

#ifdef __cplusplus
}
#endif

namespace smaug {

template <>
void EluOp<ReferenceBackend>::run() {
    auto inputs = getInput<ReferenceBackend>(Inputs);
    auto outputs = getOutput<ReferenceBackend>(Outputs);
    assert(inputs->getShape() == outputs->getShape());
    ref_elu_f32(inputs->data<float>(), outputs->data<float>(),
                 inputs->getShape().total(), alpha);
}

template <>
void SeluOp<ReferenceBackend>::run() {
    auto inputs = getInput<ReferenceBackend>(Inputs);
    auto outputs = getOutput<ReferenceBackend>(Outputs);
    assert(inputs->getShape() == outputs->getShape());
    ref_selu_f32(inputs->data<float>(), outputs->data<float>(),
                 inputs->getShape().total(), this->alpha, lambda);
}

}  // namespace smaug
