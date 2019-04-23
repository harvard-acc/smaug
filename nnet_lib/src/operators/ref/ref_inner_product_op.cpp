#include "core/backend.h"
#include "operators/common.h"
#include "operators/inner_product_op.h"
#include "utility/debug_stream.h"

#ifdef __cplusplus
extern "C" {
#endif

void ref_inner_product_nc(float* a,
                          float* b,
                          float* c,
                          int a_height,
                          int a_width,
                          int b_width,
                          int a_pad,
                          int b_pad,
                          int c_pad) {
    int input_size = a_height * (a_width + a_pad);
    int weight_size = a_width * (b_width + b_pad);
    int result_size = a_height * (b_width + c_pad);
    dmaLoad(a, a, input_size * sizeof(float));
    dmaLoad(b, b, weight_size * sizeof(float));

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
    dmaLoad(c, c, result_size * sizeof(float));
}

#ifdef __cplusplus
}
#endif

namespace smaug {

template <>
void InnerProductOp<ReferenceBackend>::run() {
    auto input = getInput(Inputs);
    auto weights = getInput(Weights);
    auto output = getOutput(Outputs);
    const TensorShape& inputShape = input->getShape();
    const TensorShape& weightShape = weights->getShape();
    const TensorShape& outputShape = output->getShape();
    assert(inputShape.getLayout() == DataLayout::NC);
    assert(weightShape.getLayout() == DataLayout::NC);
    assert(outputShape.getLayout() == DataLayout::NC);
    dout(2) << *weights << "\n";

    float* inputData = input->data<float>();
    float* weightData = weights->data<float>();
    float* outputData = output->data<float>();
    mapArrayToAccel(ref::kInnerProductHw, "a", inputData,
                    inputShape.storageSize() * sizeof(float));
    mapArrayToAccel(ref::kInnerProductHw, "b", weightData,
                    weightShape.storageSize() * sizeof(float));
    mapArrayToAccel(ref::kInnerProductHw, "c", outputData,
                    outputShape.storageSize() * sizeof(float));
    invokeKernel(ref::kInnerProductHw, ref_inner_product_nc, inputData,
                 weightData, outputData, inputShape[0], inputShape[1],
                 weightShape[1], inputShape.getPadding(1),
                 weightShape.getPadding(1), outputShape.getPadding(1));
}

}  // namespace smaug
