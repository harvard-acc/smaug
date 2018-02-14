#ifndef _CORE_SMV_H_
#define _CORE_SMV_H_

#include "core/nnet_fwd_defs.h"

void convolution3d_smv(float* a,
                       float* kernels,
                       layer_t curr_layer,
                       int kern_start,
                       float* result);

void matrix_multiply_smv(float* a,
                         float* b,
                         int a_height,
                         int b_height,
                         int b_width,
                         int a_pad,
                         activation_type act_func,
                         int result_start,
                         float* result);

#endif
