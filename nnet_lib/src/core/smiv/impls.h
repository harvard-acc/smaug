#ifndef _SMIV_IMPL_H_
#define _SMIV_IMPL_H_

#include "core/smiv/params.h"

void convolution2d_smiv_1kernel_1channel_fxp(float* a,
                                             float* kernels,
                                             int img,
                                             int kern,
                                             int chan,
                                             layer_t curr_layer,
                                             float* result);
void reduction_smiv_fxp(
        float* a, layer_t curr_layer, int img, int kern, float* result);

void matrix_multiply_with_bias_smiv_batch_fxp(float* a,
                                              float* b,
                                              int a_height,
                                              int b_height,
                                              int b_width,
                                              int a_pad,
                                              bool run_activation,
                                              float* result);

void matrix_multiply_with_bias_smiv_nobatch_fxp(float* a,
                                                float* b,
                                                int a_height,
                                                int b_height,
                                                int b_width,
                                                int a_pad,
                                                bool run_activation,
                                                float* result);

void matrix_multiply_with_bias_smiv_nobatch_vec_fxp(float* a,
                                                    float* b,
                                                    int a_height,
                                                    int b_height,
                                                    int b_width,
                                                    bool run_activation,
                                                    float* result);

#endif
