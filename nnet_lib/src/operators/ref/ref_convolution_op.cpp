#include "core/backend.h"
#include "operators/common.h"
#include "operators/convolution_op.h"

#ifdef __cplusplus
extern "C" {
#endif

void ref_conv3d_f32_nchw_valid_padding(float* input,
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
                                       int res_pad) {
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
                                int img_val = _input[img][d][i + k][j + l];
                                int kern_val = _kernels[kern][d][k][l];
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
}

void ref_conv3d_f32_nchw_same_padding(float* input,
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
                                      int res_pad) {
    const int total_row_pad = k_rows - 1;
    const int total_col_pad = k_cols - 1;
    const int left_pad = k_rows / 2;
    const int right_pad = total_col_pad - left_pad;
    const int top_pad = k_cols / 2;
    const int bottom_pad = total_row_pad - top_pad;

    // Convolution borders.
    const int start_i = -top_pad;
    const int start_j = -left_pad;
    const int end_i = img_rows - bottom_pad;
    const int end_j = img_cols - right_pad;

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
}

#ifdef __cplusplus
}
#endif

namespace smaug {

template <>
void ConvolutionOp<ReferenceBackend>::run() {
    auto input = getInput<ReferenceBackend>(Inputs);
    auto kernels = getInput<ReferenceBackend>(Kernels);
    auto output = getOutput<ReferenceBackend>(Outputs);
    const TensorShape& inputShape = input->getShape();
    const TensorShape& kernelShape = kernels->getShape();
    const TensorShape& outputShape = output->getShape();
    assert(inputShape.getLayout() == DataLayout::NCHW);
    assert(kernelShape.getLayout() == DataLayout::NCHW);
    assert(outputShape.getLayout() == DataLayout::NCHW);

    auto func = paddingType == ValidPadding
                        ? ref_conv3d_f32_nchw_valid_padding
                        : ref_conv3d_f32_nchw_same_padding;
    func(input->data<float>(),
         kernels->data<float>(),
         output->data<float>(),
         inputShape[0],
         inputShape[1],
         inputShape[2],
         inputShape[3],
         0,
         kernelShape[0],
         kernelShape[2],
         kernelShape[3],
         0,
         getRowStride(),
         getColStride(),
         outputShape[2],
         outputShape[3],
         0);
}

}  // namespace smaug
