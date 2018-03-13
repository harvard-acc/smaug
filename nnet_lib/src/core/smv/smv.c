#include "config.h"
#include "core/smv/impls.h"

void convolution3d_smv(float* a,
                       float* kernels,
                       layer_t curr_layer,
                       int kern_start,
                       float* result) {
#ifdef ENABLE_SIMD_IMPL
    convolution3d_smv_nhwc_vec_fxp(a, kernels, curr_layer, kern_start, result);
#else
    convolution3d_smv_nhwc_fxp(a, kernels, curr_layer, kern_start, result);
#endif
}

void matrix_multiply_transpose_smv(float* a,
                                   float* b,
                                   int a_height,
                                   int b_height,
                                   int b_width,
                                   int a_pad,
                                   activation_type act_func,
                                   int result_start,
                                   bool accumulate,
                                   float* result) {
    matrix_multiply_transpose_smv_nobatch_vec_fxp(
            a, b, a_height, b_height, b_width, a_pad, act_func, result_start,
            accumulate, result);
}
