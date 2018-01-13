#include <assert.h>
#include <string.h>

#include "core/ref/activation_functions.h"
#include "core/smiv/impls.h"
#include "utility/utility.h"
#include "nnet_fwd.h"

#include "smiv.h"

void reduction_smiv(float* a, layer_t curr_layer, float* result) {
#ifdef ENABLE_SIMD_IMPL
    reduction_smiv_vec_fxp(a, curr_layer, result);
#else
    reduction_smiv_fxp(a, curr_layer, result);
#endif
}

void convolution3d_smiv(float* a,
                        float* kernels,
                        layer_t curr_layer,
                        int start_chan,
                        float* result) {
#ifdef ENABLE_SIMD_IMPL
    convolution3d_smiv_1kernel_noreduce_simd_fxp(
            a, kernels, curr_layer, start_chan, result);
#else
    convolution3d_smiv_1kernel_noreduce_fxp(
        a, kernels, curr_layer, start_chan, result);
#endif
}

void matrix_multiply_with_bias_smiv(float* a,
                                    float* b,
                                    int a_height,
                                    int b_height,
                                    int b_width,
                                    int a_pad,
                                    bool run_activation,
                                    bool do_bias,
                                    float* result) {
#ifdef ENABLE_SIMD_IMPL
    matrix_multiply_with_bias_smiv_nobatch_vec_fxp(
            a, b, a_height, b_height, b_width, a_pad,
            run_activation, do_bias, result);
#else
#ifdef DISABLE_SMIV_INPUT_BATCHING
    matrix_multiply_with_bias_smiv_nobatch_fxp(
            a, b, a_height, b_height, b_width, a_pad,
            run_activation, do_bias, result);
#else
    matrix_multiply_with_bias_smiv_batch_fxp(
            a, b, a_height, b_height, b_width, a_pad,
            run_activation, do_bias, result);
#endif
#endif
}
