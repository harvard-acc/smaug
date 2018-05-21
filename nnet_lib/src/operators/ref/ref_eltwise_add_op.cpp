#include "core/backend.h"
#include "operators/common.h"
#include "operators/eltwise_add_op.h"

#ifdef __cplusplus
extern "C" {
#endif

void ref_eltwise_add_f32(float* input0,
                         float* input1,
                         float* results,
                         int input_size) {
    for (int i = 0; i < input_size; i++) {
        results[i] = input0[i] + input1[i];
    }
}

#ifdef __cplusplus
}
#endif

namespace smaug {

template <>
void EltwiseAddOp<ReferenceBackend>::run() {
    auto input0 = getInput<ReferenceBackend>(Input0);
    auto input1 = getInput<ReferenceBackend>(Input1);
    auto output = getOutput<ReferenceBackend>(Outputs);
    const TensorShape& input0Shape = input0->getShape();
    const TensorShape& input1Shape = input1->getShape();
    const TensorShape& outputShape = output->getShape();
    assert(input0Shape == input1Shape && input0Shape == outputShape);

    ref_eltwise_add_f32(input0->data<float>(),
                        input1->data<float>(),
                        output->data<float>(),
                        input0Shape.total());
}

}  // namespace smaug

