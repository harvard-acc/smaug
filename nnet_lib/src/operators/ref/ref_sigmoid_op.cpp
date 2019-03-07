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
    sigmoid_loop:
    for (int i = 0; i < input_size; i++) {
        results[i] = sigmoid_fxp(inputs[i]);
    }
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
    ref_sigmoid_f32(inputs->data<float>(), outputs->data<float>(),
                    inputs->getShape().size());
}

}  // namespace smaug
