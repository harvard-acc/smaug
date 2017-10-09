#include <assert.h>

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
    reduction_smiv_fxp(a, curr_layer, img, kern, result);
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
    // TODO: This may become an issue with the stack size.
    float temp[input_height][input_rows][input_cols + input_pad];

    // PRINT_DEBUG4D(a, input_rows, input_cols + input_pad, input_height);

    conv2d_per_image:
    for (ni = 0; ni < NUM_TEST_CASES; ni++) {
        // Loop over all inputs in this batch.
        conv2d_per_kernel:
        for (nk = 0; nk < num_kerns; nk++) {
            conv2d_per_chan:
            for (nc = 0; nc < input_height; nc++) {
                convolution2d_smiv_1kernel_1channel_fxp(
                        a, kernels, ni, nk, nc, curr_layer, &temp[0][0][0]);
            }
#ifdef ENABLE_SIMD_IMPL
            reduction_smiv_vec_fxp(&temp[0][0][0], curr_layer, ni, nk, result);
#else
            reduction_smiv(&temp[0][0][0], curr_layer, ni, nk, result);
#endif
        }
    }
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
