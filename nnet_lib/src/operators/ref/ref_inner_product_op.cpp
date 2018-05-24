#include "core/backend.h"
#include "operators/common.h"
#include "operators/inner_product_op.h"
#include "utility/debug_stream.h"

#ifdef __cplusplus
extern "C" {
#endif

void ref_inner_product_f32_nc(float* a,
                              float* b,
                              float* c,
                              int a_height,
                              int a_width,
                              int b_width,
                              int a_pad,
                              int b_pad,
                              int c_pad) {

    ARRAY_2D(float, _a, a, a_width + a_pad);
    ARRAY_2D(float, _b, b, b_width + b_pad);
    ARRAY_2D(float, _c, c, b_width + c_pad);

    for (int i = 0; i < a_height; i++) {
        for (int j = 0; j < b_width; j++) {
            float result = 0;
            for (int k = 0; k < a_width; k++) {
                float a_val = _a[i][k];
                float b_val = _b[k][j];
                result += a_val * b_val;
            }
            _c[i][j] = result;
        }
    }
}

#ifdef __cplusplus
}
#endif

namespace smaug {

template <>
void InnerProductOp<ReferenceBackend>::run() {
    auto input = getInput<ReferenceBackend>(Inputs);
    auto weights = getInput<ReferenceBackend>(Weights);
    auto output = getOutput<ReferenceBackend>(Outputs);
    const TensorShape& inputShape = input->getShape();
    const TensorShape& weightShape = weights->getShape();
    const TensorShape& outputShape = output->getShape();
    assert(inputShape.getLayout() == DataLayout::NC);
    assert(weightShape.getLayout() == DataLayout::NC);
    assert(outputShape.getLayout() == DataLayout::NC);
    dout(2) << *weights << "\n";

    ref_inner_product_f32_nc(input->data<float>(), weights->data<float>(),
                             output->data<float>(), inputShape[0],
                             inputShape[1], weightShape[1], 0, 0, 0);
}

}  // namespace smaug
