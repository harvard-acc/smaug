#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/convolution_op.h"
#include "smaug/operators/ref/ref_activation_fun_op.h"
#include "smaug/utility/debug_stream.h"

#ifdef __cplusplus
extern "C" {
#endif

/** \ingroup AladdinKernels
 *
 * A Reference implementation of a 3D convolution on NCHW data with valid
 * padding.
 */
void ref_conv3d_nchw_valid_padding(float* input,
                                   float* kernels,
                                   float* result,
                                   int img_num,
                                   int img_chans,
                                   int img_rows,
                                   int img_cols,
                                   int img_pad,
                                   int k_num,
                                   int k_rows,
                                   int k_cols,
                                   int k_pad,
                                   int k_row_stride,
                                   int k_col_stride,
                                   int res_rows,
                                   int res_cols,
                                   int res_pad,
                                   activation_type act_function,
                                   activation_param_t act_params) {
    int input_size = img_num * img_chans * img_rows * (img_cols + img_pad);
    int kernel_size = k_num * img_chans * k_rows * (k_cols + k_pad);
    int result_size = img_num * k_num * res_rows * (res_cols + res_pad);
    dmaLoad(input, input, input_size * sizeof(float));
    dmaLoad(kernels, kernels, kernel_size * sizeof(float));

    // Convolution borders.
    const int start_i = 0;
    const int start_j = 0;
    const int end_i = img_rows - k_rows + 1;
    const int end_j = img_cols - k_cols + 1;

    ARRAY_4D(float, _input, input, img_chans, img_rows, img_cols + img_pad);
    ARRAY_4D(float, _kernels, kernels, img_chans, k_rows, k_cols + k_pad);
    ARRAY_4D(float, _result, result, k_num, res_rows, res_cols + res_pad);

    conv3d_input_num:
    for (int img = 0; img < img_num; img++) {
        conv3d_kern_num:
        for (int kern = 0; kern < k_num; kern++) {
            int out_i = 0;
            conv3d_input_rows:
            for (int i = start_i; i < end_i; i += k_row_stride) {
                int out_j = 0;
                conv3d_input_cols:
                for (int j = start_j; j < end_j; j += k_col_stride) {
                    float partial_sum = 0;
                    conv3d_kernel_height:
                    // Convolution loop over the kernel.
                    for (int d = 0; d < img_chans; d++) {
                        conv3d_kernel_rows:
                        for (int k = 0; k < k_rows; k++) {
                            conv3d_kernel_cols:
                            for (int l = 0; l < k_cols; l++) {
                                float img_val = _input[img][d][i + k][j + l];
                                float kern_val = _kernels[kern][d][k][l];
                                partial_sum += img_val * kern_val;
                            }
                        }
                    }
                    _result[img][kern][out_i][out_j] = partial_sum;
                    out_j++;
                }
                out_i++;
                out_j = 0;
            }
        }
    }
    if (act_function != NO_ACTIVATION) {
        activation_fun(result, result, result_size, act_function, act_params);
    }
    dmaStore(result, result, result_size * sizeof(float));
}

/** \ingroup AladdinKernels
 *
 * A Reference implementation of a 3D convolution on NCHW data with same
 * padding.
 */
void ref_conv3d_nchw_same_padding(float* input,
                                  float* kernels,
                                  float* result,
                                  int img_num,
                                  int img_chans,
                                  int img_rows,
                                  int img_cols,
                                  int img_pad,
                                  int k_num,
                                  int k_rows,
                                  int k_cols,
                                  int k_pad,
                                  int k_row_stride,
                                  int k_col_stride,
                                  int res_rows,
                                  int res_cols,
                                  int res_pad,
                                  activation_type act_function,
                                  activation_param_t act_params) {
    int input_size = img_num * img_chans * img_rows * (img_cols + img_pad);
    int kernel_size = k_num * img_chans * k_rows * (k_cols + k_pad);
    int result_size = img_num * k_num * res_rows * (res_cols + res_pad);
    dmaLoad(input, input, input_size * sizeof(float));
    dmaLoad(kernels, kernels, kernel_size * sizeof(float));

    const int total_row_pad = k_rows - 1;
    const int total_col_pad = k_cols - 1;
    const int left_pad = k_rows / 2;
    const int right_pad = total_col_pad - left_pad;
    const int top_pad = k_cols / 2;
    const int bottom_pad = total_row_pad - top_pad;

    // Convolution borders.
    const int start_i = -top_pad;
    const int start_j = -left_pad;
    const int end_i = img_rows + bottom_pad - k_rows + 1;
    const int end_j = img_cols + right_pad - k_cols + 1;

    ARRAY_4D(float, _input, input, img_chans, img_rows, img_cols + img_pad);
    ARRAY_4D(float, _kernels, kernels, img_chans, k_rows, k_cols + k_pad);
    ARRAY_4D(float, _result, result, k_num, res_rows, res_cols + res_pad);

    conv3d_input_num:
    for (int img = 0; img < img_num; img++) {
        conv3d_kern_num:
        for (int kern = 0; kern < k_num; kern++) {
            int out_i = 0;
            conv3d_input_rows:
            for (int i = start_i; i < end_i; i += k_row_stride) {
                int out_j = 0;
                conv3d_input_cols:
                for (int j = start_j; j < end_j; j += k_col_stride) {
                    float partial_sum = 0;

                    conv3d_kernel_height:
                    // Convolution loop over the kernel.
                    for (int d = 0; d < img_chans; d++) {
                        conv3d_kernel_rows:
                        for (int k = 0; k < k_rows; k++) {
                            bool rowInBounds =
                                    (i + k) >= 0 && (i + k) < img_rows;
                            conv3d_kernel_cols:
                            for (int l = 0; l < k_cols; l++) {
                                bool colInBounds =
                                        (j + l) >= 0 && (j + l) < img_cols;
                                float img_val = rowInBounds && colInBounds
                                                ? _input[img][d][i + k][j + l]
                                                : 0;
                                float kern_val = rowInBounds && colInBounds
                                                       ? _kernels[kern][d][k][l]
                                                       : 0;
                                partial_sum += img_val * kern_val;
                            }
                        }
                    }
                    _result[img][kern][out_i][out_j] = partial_sum;
                    out_j++;
                }
                out_i++;
                out_j = 0;
            }
        }
    }
    if (act_function != NO_ACTIVATION) {
        activation_fun(result, result, result_size, act_function, act_params);
    }
    dmaStore(result, result, result_size * sizeof(float));
}

/** \ingroup AladdinKernels
 *
 * A Reference implementation of a 3D convolution on NHWC data with valid
 * padding.
 */
void ref_conv3d_nhwc_valid_padding(float* input,
                                   float* kernels,
                                   float* result,
                                   int img_num,
                                   int img_chans,
                                   int img_rows,
                                   int img_cols,
                                   int img_pad,
                                   int k_num,
                                   int k_rows,
                                   int k_cols,
                                   int k_pad,
                                   int k_row_stride,
                                   int k_col_stride,
                                   int res_rows,
                                   int res_cols,
                                   int res_pad,
                                   activation_type act_function,
                                   activation_param_t act_params) {
    int input_size = img_num * img_rows * img_cols * (img_chans + img_pad);
    int kernel_size = k_num * k_rows * k_cols * (img_chans + k_pad);
    int result_size = img_num * res_rows * res_cols * (k_num + res_pad);
    dmaLoad(input, input, input_size * sizeof(float));
    dmaLoad(kernels, kernels, kernel_size * sizeof(float));

    // Convolution borders.
    const int start_i = 0;
    const int start_j = 0;
    const int end_i = img_rows - k_rows + 1;
    const int end_j = img_cols - k_cols + 1;

    ARRAY_4D(float, _input, input, img_rows, img_cols, img_chans + img_pad);
    ARRAY_4D(float, _kernels, kernels, k_rows, k_cols, img_chans + k_pad);
    ARRAY_4D(float, _result, result, res_rows, res_cols, k_num + res_pad);

    conv3d_input_num:
    for (int img = 0; img < img_num; img++) {
        conv3d_kern_num:
        for (int kern = 0; kern < k_num; kern++) {
            int out_i = 0;
            conv3d_input_rows:
            for (int i = start_i; i < end_i; i += k_row_stride) {
                int out_j = 0;
                conv3d_input_cols:
                for (int j = start_j; j < end_j; j += k_col_stride) {
                    float partial_sum = 0;
                    conv3d_kernel_height:
                    // Convolution loop over the kernel.
                    for (int d = 0; d < img_chans; d++) {
                        conv3d_kernel_rows:
                        for (int k = 0; k < k_rows; k++) {
                            conv3d_kernel_cols:
                            for (int l = 0; l < k_cols; l++) {
                                float img_val = _input[img][i + k][j + l][d];
                                float kern_val = _kernels[kern][k][l][d];
                                partial_sum += img_val * kern_val;
                            }
                        }
                    }
                    _result[img][out_i][out_j][kern] = partial_sum;
                    out_j++;
                }
                out_i++;
                out_j = 0;
            }
        }
    }
    if (act_function != NO_ACTIVATION) {
        activation_fun(result, result, result_size, act_function, act_params);
    }
    dmaStore(result, result, result_size * sizeof(float));
}

/** \ingroup AladdinKernels
 *
 * A Reference implementation of a 3D convolution on NHWC data with same
 * padding.
 */
void ref_conv3d_nhwc_same_padding(float* input,
                                  float* kernels,
                                  float* result,
                                  int img_num,
                                  int img_chans,
                                  int img_rows,
                                  int img_cols,
                                  int img_pad,
                                  int k_num,
                                  int k_rows,
                                  int k_cols,
                                  int k_pad,
                                  int k_row_stride,
                                  int k_col_stride,
                                  int res_rows,
                                  int res_cols,
                                  int res_pad,
                                  activation_type act_function,
                                  activation_param_t act_params) {
    int input_size = img_num * img_rows * img_cols * (img_chans + img_pad);
    int kernel_size = k_num * k_rows * k_cols * (img_chans + k_pad);
    int result_size = img_num * res_rows * res_cols * (k_num + res_pad);
    dmaLoad(input, input, input_size * sizeof(float));
    dmaLoad(kernels, kernels, kernel_size * sizeof(float));

    const int total_row_pad = k_rows - 1;
    const int total_col_pad = k_cols - 1;
    const int left_pad = k_rows / 2;
    const int right_pad = total_col_pad - left_pad;
    const int top_pad = k_cols / 2;
    const int bottom_pad = total_row_pad - top_pad;

    // Convolution borders.
    const int start_i = -top_pad;
    const int start_j = -left_pad;
    const int end_i = img_rows + bottom_pad - k_rows + 1;
    const int end_j = img_cols + right_pad - k_cols + 1;

    ARRAY_4D(float, _input, input, img_rows, img_cols, img_chans + img_pad);
    ARRAY_4D(float, _kernels, kernels, k_rows, k_cols, img_chans + k_pad);
    ARRAY_4D(float, _result, result, res_rows, res_cols, k_num + res_pad);

    conv3d_input_num:
    for (int img = 0; img < img_num; img++) {
        conv3d_kern_num:
        for (int kern = 0; kern < k_num; kern++) {
            int out_i = 0;
            conv3d_input_rows:
            for (int i = start_i; i < end_i; i += k_row_stride) {
                int out_j = 0;
                conv3d_input_cols:
                for (int j = start_j; j < end_j; j += k_col_stride) {
                    float partial_sum = 0;

                    conv3d_kernel_height:
                    // Convolution loop over the kernel.
                    for (int d = 0; d < img_chans; d++) {
                        conv3d_kernel_rows:
                        for (int k = 0; k < k_rows; k++) {
                            bool rowInBounds =
                                    (i + k) >= 0 && (i + k) < img_rows;
                            conv3d_kernel_cols:
                            for (int l = 0; l < k_cols; l++) {
                                bool colInBounds =
                                        (j + l) >= 0 && (j + l) < img_cols;
                                float img_val = rowInBounds && colInBounds
                                                ? _input[img][i + k][j + l][d]
                                                : 0;
                                float kern_val = rowInBounds && colInBounds
                                                       ? _kernels[kern][k][l][d]
                                                       : 0;
                                partial_sum += img_val * kern_val;
                            }
                        }
                    }
                    _result[img][out_i][out_j][kern] = partial_sum;
                    out_j++;
                }
                out_i++;
                out_j = 0;
            }
        }
    }
    if (act_function != NO_ACTIVATION) {
        activation_fun(result, result, result_size, act_function, act_params);
    }
    dmaStore(result, result, result_size * sizeof(float));
}

#ifdef __cplusplus
}
#endif

namespace smaug {

template <>
void ConvolutionOp<ReferenceBackend>::run() {
    auto input = getInput(Inputs);
    auto kernels = getInput(Kernels);
    auto output = getOutput(Outputs);
    const TensorShape& inputShape = input->getShape();
    const TensorShape& kernelShape = kernels->getShape();
    const TensorShape& outputShape = output->getShape();
    dout(2) << *kernels << "\n";

    float* inputData = input->data<float>();
    float* kernelData = kernels->data<float>();
    float* outputData = output->data<float>();
    mapArrayToAccel(ref::kConvolutionHw, "input", inputData,
                    inputShape.storageSize() * sizeof(float));
    mapArrayToAccel(ref::kConvolutionHw, "kernels", kernelData,
                    kernelShape.storageSize() * sizeof(float));
    mapArrayToAccel(ref::kConvolutionHw, "result", outputData,
                    outputShape.storageSize() * sizeof(float));
    bool isNCHW = input->getShape().getLayout() == NCHW;
    auto func = isNCHW ? (paddingType == ValidPadding
                                  ? ref_conv3d_nchw_valid_padding
                                  : ref_conv3d_nchw_same_padding)
                       : (paddingType == ValidPadding
                                  ? ref_conv3d_nhwc_valid_padding
                                  : ref_conv3d_nhwc_same_padding);
    int rowIdx = isNCHW ? 2 : 1;
    int colIdx = isNCHW ? 3 : 2;
    int chanIdx = isNCHW ? 1 : 3;
    invokeKernel(ref::kConvolutionHw, func, inputData, kernelData, outputData,
                 inputShape[0], inputShape[chanIdx], inputShape[rowIdx],
                 inputShape[colIdx], inputShape.getPadding(3), kernelShape[0],
                 kernelShape[rowIdx], kernelShape[colIdx],
                 kernelShape.getPadding(3), getRowStride(), getColStride(),
                 outputShape[rowIdx], outputShape[colIdx],
                 outputShape.getPadding(3), actInfo.function, actInfo.params);
}

}  // namespace smaug
