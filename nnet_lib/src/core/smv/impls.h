#ifndef _SMV_IMPL_H_
#define _SMV_IMPL_H_

#include "core/nnet_fwd_defs.h"
#include "core/smiv/params.h"
#include "core/smv/params.h"

void convolution3d_smv_nhwc_fxp(float* a,
                                float* kernels,
                                layer_t curr_layer,
                                int kern_start,
                                float* result);

void convolution3d_smv_nhwc_vec_fxp(float* a,
                                    float* kernels,
                                    layer_t curr_layer,
                                    int kern_start,
                                    float* result);

void matrix_multiply_transpose_smv_nobatch_vec_fxp(
        float* a,
        float* b,
        int a_height,
        int b_width,
        int b_height,
        int a_pad,
        activation_type act_func,
        int result_start,
        float* result);

#endif
