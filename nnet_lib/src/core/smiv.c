#include <assert.h>
#include <string.h>

#include "core/activation_functions.h"
#include "core/smiv/impls.h"
#include "utility/utility.h"
#include "nnet_fwd.h"

#include "smiv.h"

void reduction_smiv(float *a,
                    layer_t curr_layer,
                    int img,
                    int kern,
                    float *result) {
#ifdef ENABLE_SIMD_IMPL
    reduction_smiv_vec_fxp(a, curr_layer, img, kern, result);
#else
    reduction_smiv_fxp(a, curr_layer, img, kern, result);
#endif
}

void convolution2d_smiv(float* a,
                        float* kernels,
                        layer_t curr_layer,
                        float* result) {
    int ni, nk, nc;

    const int input_height = curr_layer.inputs.height;
    const int input_rows = curr_layer.inputs.rows;
    const int input_cols = curr_layer.inputs.cols;
    const int input_pad = curr_layer.inputs.align_pad;
    const int num_kerns = curr_layer.outputs.height;

    // Stores the unreduced convolution output.
    // TODO: Obviously this doesn't work for HW!
    float* temp = (float*)malloc(input_height * input_rows *
                                 (input_cols + input_pad) * sizeof(float));
    memset(temp,
           0,
           input_height * input_rows * (input_cols + input_pad) *
                   sizeof(float));

    // PRINT_DEBUG4D(a, input_rows, input_cols + input_pad, input_height);

    conv2d_per_image:
    for (ni = 0; ni < NUM_TEST_CASES; ni++) {
        // Loop over all inputs in this batch.
        conv2d_per_kernel:
        for (nk = 0; nk < num_kerns; nk++) {
            conv2d_per_chan:
            for (nc = 0; nc < input_height; nc++) {
#ifdef ENABLE_SIMD_IMPL
                convolution2d_smiv_1kernel_1channel_simd_fxp(
                        a, kernels, ni, nk, nc, curr_layer, temp);
#else
                convolution2d_smiv_1kernel_1channel_fxp(
                        a, kernels, ni, nk, nc, curr_layer, temp);
#endif
            }
            reduction_smiv(temp, curr_layer, ni, nk, result);
        }
    }
    free(temp);
}

void matrix_multiply_with_bias_smiv(float* a,
                                    float* b,
                                    int a_height,
                                    int b_height,
                                    int b_width,
                                    int a_pad,
                                    bool run_activation,
                                    float* result) {
#ifdef ENABLE_SIMD_IMPL
    matrix_multiply_with_bias_smiv_nobatch_vec_fxp(
            a, b, a_height, b_height, b_width, a_pad, run_activation, result);
#else
#ifdef DISABLE_SMIV_INPUT_BATCHING
    matrix_multiply_with_bias_smiv_nobatch_fxp(
            a, b, a_height, b_height, b_width, a_pad, run_activation, result);
#else
    matrix_multiply_with_bias_smiv_batch_fxp(
            a, b, a_height, b_height, b_width, a_pad, run_activation, result);
#endif
#endif
}
