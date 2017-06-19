#include "core/pooling.h"
#include "core/zeropad.h"
#include "utility/utility.h"
#include "nnet_fwd.h"

#include "convolution.h"

// Perform a 2D convolution on the data in @input with zero padding.
//
// The amount of padding is determined by the convolution operation configured
// in the @curr_layer struct.
//
// NOTE: The data is not actually placed in @result, but rather into @input!
// This is because the input data is zero-padded and copied into result. To
// avoid a second copy back into @input, we simply execute the convolution using
// @result as the input array instead.
void convolution2d_zeropad(float* input,
                           float* kernels,
                           layer_t curr_layer,
                           float* result) {
    int padding = (curr_layer.field_size - 1) / 2;
    copy_zeropad(input, curr_layer, padding, result);
    PRINT_MSG("After zeropadding:\n");
    PRINT_DEBUG4D(result,
                  curr_layer.input_rows,
                  curr_layer.input_cols,
                  curr_layer.input_height);
    convolution2d_no_padding(result, kernels, curr_layer, input);
}

// Perform a 2D convolution operation over the data in @a with all kernels.
//
// The convolutions are specified through the curr_layers struct. Zero padding
// for borders is not handled.
//
// @a contains a stack of 3D images, and kernels contains a stack of 3D
// kernels, whose dimensions are specified in @curr_layer.
//
// The result of the operation is placed into @result, which is assumed to be
// large enough to hold the data. @result is also a stack of 3D images.
void convolution2d_no_padding(float* a,
                              float* kernels,
                              layer_t curr_layer,
                              float* result) {

    int ni, nk;

conv2d_per_image:
    for (ni = 0; ni < NUM_TEST_CASES; ni++) {
        // Loop over all inputs in this batch.
    conv2d_per_kernel:
        for (nk = 0; nk < curr_layer.output_height; nk++) {
            convolution2d_kernel_no_padding(a, kernels, ni, nk, curr_layer, result);
        }
    }
}

// Perform a 2D convolution over one 3D image and one 3D kernel.
void convolution2d_kernel_no_padding(float* a,
                                     float* kernels,
                                     int img,
                                     int kern,
                                     layer_t curr_layer,
                                     float* result) {
    int d, i, j, k, l;

    const int a_height = curr_layer.input_rows;
    const int a_width = curr_layer.input_cols + curr_layer.input_data_align_pad;

    const int result_height = curr_layer.output_rows;
    const int result_width =
            curr_layer.output_cols + curr_layer.output_data_align_pad;

    // Filter is k_width x k_width x k_height.
    const int k_width = curr_layer.field_size;
    const int k_height =  curr_layer.input_height;
    const int k_stride = curr_layer.field_stride;
    const int num_kerns = curr_layer.output_height;

    // Convolution borders.
    const int start_i = 0;
    const int start_j = 0;
    const int end_i = curr_layer.output_cols;
    const int end_j = result_height;

    float partial_sum, a_val, kern_val;

    ARRAY_4D(float, _a, a, k_height, a_height, a_width);
    ARRAY_4D(float, _kernels, kernels, k_height, k_width, k_width);
    ARRAY_4D(float, _result, result, num_kerns, result_height, result_width);

conv2d_input_rows:
    // Convolution loop over the output pixels in this depth slice (kern).
    for (i = start_i; i < end_i; i+= k_stride) {
    conv2d_input_cols:
        for (j = start_j; j < end_j; j+= k_stride) {
            partial_sum = 0;
        conv2d_kernel_height:
            // Convolution loop over the kernel.
            for (d = 0; d < k_height; d++) {
            conv2d_kernel_rows:
                for (k = 0; k < k_width; k++) {
                conv2d_kernel_cols:
                    for (l = 0; l < k_width; l++) {
                        a_val = conv_float2fixed(_a[img][d][i+k][j+l]);
                        kern_val = conv_float2fixed(_kernels[kern][d][k][l]);
                        partial_sum += conv_float2fixed(a_val * kern_val);
                    }
                }
            }
            _result[img][kern][i][j] = partial_sum;
        }
    }
}
