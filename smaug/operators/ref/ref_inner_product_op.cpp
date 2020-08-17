#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/inner_product_op.h"
#include "smaug/operators/ref/ref_activation_fun_op.h"
#include "smaug/utility/debug_stream.h"

#ifdef __cplusplus
extern "C" {
#endif

/** \ingroup AladdinKernels
 *
 * A Reference implementation of an inner product operator: `C = A x B`.
 *
 * @param a A matrix of dimensions a_height x a_width
 * @param b A matrix of dimensions a_width x b_width
 * @param c A matrix of dimensions a_height x b_width
 * @param a_height Number of rows in A
 * @param a_width Number of columns in A
 * @param b_width Number of columns in B
 * @param a_pad Additional alignment zero-padding on a.
 * @param b_pad Additional alignment zero-padding on b.
 * @param c_pad Additional alignment zero-padding on c.
 * @param act_function The activation function to apply on the result of the
 * inner product.
 * @param act_params Parameters to the activation function.
 */
void ref_inner_product_ab_times_bc(float* a,
                                   float* b,
                                   float* c,
                                   int a_height,
                                   int a_width,
                                   int b_width,
                                   int a_pad,
                                   int b_pad,
                                   int c_pad,
                                   activation_type act_function,
                                   activation_param_t act_params) {
    int input_size = a_height * (a_width + a_pad);
    int weight_size = a_width * (b_width + b_pad);
    int result_size = a_height * (b_width + c_pad);
    dmaLoad(a, a, input_size * sizeof(float));
    dmaLoad(b, b, weight_size * sizeof(float));

    ARRAY_2D(float, _a, a, a_width + a_pad);
    ARRAY_2D(float, _b, b, b_width + b_pad);
    ARRAY_2D(float, _c, c, b_width + c_pad);

    matmul0:
    for (int i = 0; i < a_height; i++) {
        matmul1:
        for (int j = 0; j < b_width; j++) {
            float result = 0;
            matmul2:
            for (int k = 0; k < a_width; k++) {
                float a_val = _a[i][k];
                float b_val = _b[k][j];
                result += a_val * b_val;
            }
            _c[i][j] = result;
        }
    }
    if (act_function != NO_ACTIVATION) {
        activation_fun(c, c, result_size, act_function, act_params);
    }
    dmaLoad(c, c, result_size * sizeof(float));
}

/** \ingroup AladdinKernels
 *
 * A Reference implementation of an inner product operator. `C = A x *
 * B_transpose`.
 *
 * @param a A matrix of dimensions a_height x b_width
 * @param b A matrix of dimensions b_height x b_width
 * @param c A matrix of dimensions a_height x b_width
 * @param a_height Number of rows in A
 * @param b_width Number of columns in B
 * @param b_height Number of rows in B
 * @param a_pad Additional alignment zero-padding on a.
 * @param b_pad Additional alignment zero-padding on b.
 * @param c_pad Additional alignment zero-padding on c.
 * @param act_function The activation function to apply on the result of the
 * inner product.
 * @param act_params Parameters to the activation function.
 */
void ref_inner_product_ab_times_cb(float* a,
                                   float* b,
                                   float* c,
                                   int a_height,
                                   int b_width,
                                   int b_height,
                                   int a_pad,
                                   int b_pad,
                                   int c_pad,
                                   activation_type act_function,
                                   activation_param_t act_params) {
    int a_width = b_width;
    int input_size = a_height * (a_width + a_pad);
    int weight_size = b_height * (b_width + b_pad);
    int result_size = a_height * (b_height + c_pad);
    dmaLoad(a, a, input_size * sizeof(float));
    dmaLoad(b, b, weight_size * sizeof(float));

    ARRAY_2D(float, _a, a, a_width);
    ARRAY_2D(float, _b, b, b_width);
    ARRAY_2D(float, _c, c, b_height);

    matmul0:
    for (int i = 0; i < a_height; i++) {
        matmul1:
        for (int j = 0; j < b_height; j++) {
            float result = 0;
            matmul2:
            for (int k = 0; k < a_width; k++) {
                float a_val = _a[i][k];
                float b_val = _b[j][k];
                result += a_val * b_val;
            }
            _c[i][j] = result;
        }
    }
    if (act_function != NO_ACTIVATION) {
        activation_fun(c, c, result_size, act_function, act_params);
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
    assert(weightShape.getLayout() == DataLayout::NC ||
           weightShape.getLayout() == DataLayout::CN);
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
    bool weightsTransposed = weightShape.getLayout() == DataLayout::NC;
    auto func = weightsTransposed ? ref_inner_product_ab_times_cb
                                  : ref_inner_product_ab_times_bc;
    int actIdx = weightsTransposed ? 1 : 0;
    int neuronIdx = weightsTransposed ? 0 : 1;
    invokeKernel(ref::kInnerProductHw, func, inputData, weightData, outputData,
                 inputShape[0], weightShape[actIdx], weightShape[neuronIdx],
                 inputShape.getPadding(1), weightShape.getPadding(1),
                 outputShape.getPadding(1), actInfo.function, actInfo.params);
}

}  // namespace smaug
