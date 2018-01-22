#ifndef _SMIV_IMPL_H_
#define _SMIV_IMPL_H_

#include "core/smiv/params.h"

void convolution3d_smiv_1kernel_noreduce_fxp(float* a,
                                             float* kernels,
                                             layer_t curr_layer,
                                             int start_chan,
                                             float* result);

void convolution3d_smiv_1kernel_noreduce_simd_fxp(float* a,
                                                  float* kernels,
                                                  layer_t curr_layer,
                                                  int start_chan,
                                                  float* result);

void reduction_smiv_fxp(float* a, layer_t curr_layer, float* result);

void reduction_smiv_vec_fxp(float* a, layer_t curr_layer, float* result);

void matrix_multiply_with_bias_smiv_batch_fxp(float* a,
                                              float* b,
                                              int a_height,
                                              int b_height,
                                              int b_width,
                                              int a_pad,
                                              activation_type act_func,
                                              bool do_bias,
                                              float* result);

void matrix_multiply_with_bias_smiv_nobatch_fxp(float* a,
                                                float* b,
                                                int a_height,
                                                int b_height,
                                                int b_width,
                                                int a_pad,
                                                activation_type act_func,
                                                bool do_bias,
                                                float* result);

void matrix_multiply_with_bias_smiv_nobatch_vec_fxp(float* a,
                                                    float* b,
                                                    int a_height,
                                                    int b_height,
                                                    int b_width,
                                                    int a_pad,
                                                    activation_type act_func,
                                                    bool do_bias,
                                                    float* result);

#endif
