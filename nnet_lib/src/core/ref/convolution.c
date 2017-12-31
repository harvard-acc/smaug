#include "core/ref/convolution.h"
#include "core/ref/pooling.h"
#include "core/ref/zeropad.h"
#include "utility/utility.h"
#include "nnet_fwd.h"


// Perform a 3D convolution on the data in @input with zero padding.
//
// The amount of padding is determined by the convolution operation configured
// in the @curr_layer struct.
//
// NOTE: The data is not actually placed in @result, but rather into @input!
// This is because the input data is zero-padded and copied into result. To
// avoid a second copy back into @input, we simply execute the convolution using
// @result as the input array instead.
void convolution3d_zeropad(float* input,
                           float* kernels,
                           layer_t* layers,
                           int lnum,
                           float* result) {
    layer_t curr_layer = layers[lnum];
    copy_zeropad(input, layers, lnum, result);
    PRINT_MSG("After zeropadding:\n");
    PRINT_DEBUG4D(result,
                  curr_layer.inputs.rows,
                  curr_layer.inputs.cols + curr_layer.inputs.align_pad,
                  curr_layer.inputs.height);
    convolution3d_no_padding(result, kernels, curr_layer, input);
}

void convolution2d_depthwise_zeropad(float* input,
                                   float* kernels,
                                   layer_t* layers,
                                   int lnum,
                                   float* result) {
    layer_t curr_layer = layers[lnum];
    copy_zeropad(input, layers, lnum, result);
    PRINT_MSG("After zeropadding:\n");
    PRINT_DEBUG4D(result,
                  curr_layer.inputs.rows,
                  curr_layer.inputs.cols + curr_layer.inputs.align_pad,
                  curr_layer.inputs.height);
    convolution2d_depthwise_nopadding(result, kernels, curr_layer, input);
}

// Perform a 3D convolution operation over the data in @a with all kernels.
//
// The convolutions are specified through the curr_layers struct. Zero padding
// for borders is not handled.
//
// @a contains a stack of 3D images, and kernels contains a stack of 3D
// kernels, whose dimensions are specified in @curr_layer.
//
// The result of the operation is placed into @result, which is assumed to be
// large enough to hold the data. @result is also a stack of 3D images.
void convolution3d_no_padding(float* a,
                              float* kernels,
                              layer_t curr_layer,
                              float* result) {

    int ni, nk;

    conv2d_per_image:
    for (ni = 0; ni < NUM_TEST_CASES; ni++) {
        // Loop over all inputs in this batch.
        conv2d_per_kernel:
        for (nk = 0; nk < curr_layer.outputs.height; nk++) {
            convolution3d_kernel_no_padding(a, kernels, ni, nk, curr_layer, result);
        }
    }
}

void convolution2d_depthwise_nopadding(float* a,
                                       float* kernels,
                                       layer_t curr_layer,
                                       float* result) {
    conv_depthwise_per_image:
    for (int ni = 0; ni < NUM_TEST_CASES; ni++) {
        conv_depthwise2d_per_kernel:
        for (int nk = 0; nk < curr_layer.inputs.height; nk++) {
            convolution2d_depthwise_single_kernel(
                    a, kernels, ni, nk, curr_layer, result);
        }
    }
}

void convolution3d_pointwise_nopadding(float* a,
                                       float* kernels,
                                       layer_t curr_layer,
                                       float* result) {
    conv_depthwise_per_image:
    for (int ni = 0; ni < NUM_TEST_CASES; ni++) {
        conv_depthwise2d_per_kernel:
        for (int nk = 0; nk < curr_layer.outputs.height; nk++) {
            convolution3d_pointwise_direct(
                    a, kernels, ni, nk, curr_layer, result);
        }
    }
}

// Perform a 3D convolution over one 3D image and one 3D kernel.
void convolution3d_kernel_no_padding(float* a,
                                     float* kernels,
                                     int img,
                                     int kern,
                                     layer_t curr_layer,
                                     float* result) {
    int d, i, j, k, l;

    const int a_height = curr_layer.inputs.rows;
    const int a_width = curr_layer.inputs.cols + curr_layer.inputs.align_pad;

    const int result_height = curr_layer.outputs.rows;
    const int result_width =
            curr_layer.outputs.cols + curr_layer.outputs.align_pad;

    // Filter is k_width x k_width x k_height.
    const int k_width = curr_layer.weights.cols;
    const int k_height =  curr_layer.inputs.height;
    const int k_stride = curr_layer.field_stride;
    const int k_pad = curr_layer.weights.align_pad;
    const int num_kerns = curr_layer.outputs.height;

    // Convolution borders.
    const int start_i = 0;
    const int start_j = 0;
    const int end_i = curr_layer.outputs.rows;
    const int end_j = result_width;

    float partial_sum, a_val, kern_val;

    ARRAY_4D(float, _a, a, k_height, a_height, a_width);
    ARRAY_4D(float, _kernels, kernels, k_height, k_width, k_width + k_pad);
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

// Applies a 2D filter from weights channel @chan on inputs channel n.
void convolution2d_depthwise_single_kernel(float* a,
                                           float* kernels,
                                           int img,
                                           int chan,
                                           layer_t curr_layer,
                                           float* result) {
    const int a_rows = curr_layer.inputs.rows;
    const int a_cols = curr_layer.inputs.cols + curr_layer.inputs.align_pad;
    const int a_height = curr_layer.inputs.height;

    const int result_rows = curr_layer.outputs.rows;
    const int result_cols =
            curr_layer.outputs.cols + curr_layer.outputs.align_pad;

    // Filter is k_cols x k_cols x k_height.
    const int k_cols = curr_layer.weights.cols;
    const int k_stride = curr_layer.field_stride;
    const int k_pad = curr_layer.weights.align_pad;

    // Convolution borders.
    const int start_i = 0;
    const int start_j = 0;
    const int end_i = curr_layer.outputs.rows;
    const int end_j = result_cols;

    float partial_sum, a_val, kern_val;

    ARRAY_4D(float, _a, a, a_height, a_rows, a_cols);
    // For depthwise conv, each filter has height 1, so there are only 3
    // dimensions.
    ARRAY_3D(float, _kernels, kernels, k_cols, k_cols + k_pad);
    // Each result has the same height as the input.
    ARRAY_4D(float, _result, result, a_height, result_rows, result_cols);

    conv2d_input_rows:
    for (int i = start_i; i < end_i; i+= k_stride) {
        conv2d_input_cols:
        for (int j = start_j; j < end_j; j+= k_stride) {
            partial_sum = 0;
            conv2d_kernel_rows:
            for (int k = 0; k < k_cols; k++) {
                conv2d_kernel_cols:
                for (int l = 0; l < k_cols; l++) {
                    a_val = conv_float2fixed(_a[img][chan][i+k][j+l]);
                    kern_val = conv_float2fixed(_kernels[chan][k][l]);
                    partial_sum += conv_float2fixed(a_val * kern_val);
                }
            }
            _result[img][chan][i][j] = partial_sum;
        }
    }
}

void convolution3d_pointwise_direct(float* a,
                                    float* kernels,
                                    int img,
                                    int kern,
                                    layer_t curr_layer,
                                    float* result) {
    const int a_rows = curr_layer.inputs.rows;
    const int a_cols = curr_layer.inputs.cols + curr_layer.inputs.align_pad;
    const int a_height = curr_layer.inputs.height;

    const int result_height = curr_layer.outputs.height;
    const int result_rows = curr_layer.outputs.rows;
    const int result_cols =
            curr_layer.outputs.cols + curr_layer.outputs.align_pad;

    // Filter is 1 x 1 x k_cols.
    const int k_cols = curr_layer.inputs.height;
    const int k_stride = curr_layer.field_stride;
    const int k_pad = curr_layer.weights.align_pad;
    const int num_kerns = curr_layer.outputs.height;

    // Convolution borders.
    const int start_i = 0;
    const int start_j = 0;
    const int end_i = curr_layer.outputs.rows;
    const int end_j = result_cols;

    float partial_sum, a_val, kern_val;

    ARRAY_4D(float, _a, a, a_height, a_rows, a_cols);
    // For pointwise conv, each filter is 1x1xH, so collectively there are only
    // two dimensions that matter.
    ARRAY_2D(float, _kernels, kernels, k_cols + k_pad);
    ARRAY_4D(float, _result, result, result_height, result_rows, result_cols);


    conv_pw_input_rows:
    for (int i = start_i; i < end_i; i+= k_stride) {
        conv_pw_input_cols:
        for (int j = start_j; j < end_j; j+= k_stride) {
            partial_sum = 0;
            conv_pw_kernel_rows:
            for (int k = 0; k < k_cols ; k++) {
                a_val = conv_float2fixed(_a[img][k][i][j]);
                kern_val = conv_float2fixed(_kernels[kern][k]);
                partial_sum += conv_float2fixed(a_val * kern_val);
            }
            _result[img][kern][i][j] = partial_sum;
        }
    }
}
