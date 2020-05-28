#ifndef _OPERATORS_SMV_KERNELS_H_
#define _OPERATORS_SMV_KERNELS_H_

#include "smaug/operators/common.h"

#ifdef __cplusplus
extern "C" {
#endif

void smv_conv3d_nhwc_vec_fxp(float16* host_inputs,
                             float16* host_weights,
                             float16* host_results,
                             float* inputs,
                             float* weights,
                             float* results,
                             int inputs_dims[4],
                             int weights_dims[4],
                             int results_dims[4],
                             int inputs_align_pad,
                             int weights_pad,
                             int results_pad,
                             int inputs_halo_pad[4],
                             int row_stride,
                             int col_stride,
                             int ifmap_start,
                             int kern_start,
                             bool accumulate,
                             bool read_inputs,
                             bool read_weights,
                             bool send_results,
                             activation_type act_function,
                             activation_param_t act_params,
                             SamplingInfo* sampling);

void smv_matrix_multiply_transpose_nc_vec_fxp(float16* host_a,
                                              float16* host_b,
                                              float16* host_results,
                                              float* a,
                                              float* b,
                                              float* results,
                                              int a_dims[2],
                                              int b_dims[2],
                                              int results_dims[2],
                                              int a_pad,
                                              int b_pad,
                                              int results_pad,
                                              int a_start,
                                              int result_start,
                                              bool accumulate,
                                              bool read_inputs,
                                              bool send_results,
                                              activation_type act_function,
                                              activation_param_t act_params,
                                              SamplingInfo* sampling);

void smv_maxpooling_nhwc_vec_fxp(float16* host_inputs,
                                 float16* host_results,
                                 float* inputs,
                                 float* results,
                                 int inputs_dims[4],
                                 int results_dims[4],
                                 int inputs_pad,
                                 int results_pad,
                                 int pool_rows,
                                 int pool_cols,
                                 int row_stride,
                                 int col_stride,
                                 int ofmap_start,
                                 SamplingInfo* sampling);

void smv_avgpooling_nhwc_vec_fxp(float16* host_inputs,
                                 float16* host_results,
                                 float* inputs,
                                 float* results,
                                 int inputs_dims[4],
                                 int results_dims[4],
                                 int inputs_pad,
                                 int results_pad,
                                 int pool_rows,
                                 int pool_cols,
                                 int row_stride,
                                 int col_stride,
                                 int ofmap_start,
                                 SamplingInfo* sampling);

void smv_batch_norm_post_fc_nc_vec_fxp(float16* host_inputs,
                                       float16* host_weights,
                                       float16* host_results,
                                       float* inputs,
                                       float* weights,
                                       float* results,
                                       int inputs_dims[2],
                                       int weights_acts,
                                       int inputs_pad,
                                       int inputs_start,
                                       int send_results,
                                       activation_type act_function,
                                       activation_param_t act_params);

void smv_batch_norm_post_conv_nchw_vec_fxp(float16* host_inpits,
                                           float16* host_weights,
                                           float16* host_results,
                                           float* inputs,
                                           float* weights,
                                           float* results,
                                           int inputs_dims[4],
                                           int weights_chans,
                                           int inputs_pad,
                                           int weights_pad,
                                           int weights_start,
                                           activation_type act_function,
                                           activation_param_t act_params);

void smv_batch_norm_post_conv_nhwc_vec_fxp(float16* host_inpits,
                                           float16* host_weights,
                                           float16* host_results,
                                           float* inputs,
                                           float* weights,
                                           float* results,
                                           int inputs_dims[4],
                                           int weights_chans,
                                           int inputs_pad,
                                           int weights_pad,
                                           int weights_start,
                                           activation_type act_function,
                                           activation_param_t act_params,
                                           SamplingInfo* sampling);

void smv_activation_fun_nc_vec_fxp(float16* host_inputs,
                                   float16* host_results,
                                   float* inputs,
                                   float* results,
                                   int inputs_size,
                                   activation_type function,
                                   activation_param_t params);

void smv_softmax_nc_vec_fxp(float16* host_inputs,
                            float16* host_results,
                            float* inputs,
                            float* results,
                            int input_num,
                            int input_size,
                            int input_pad);

void smv_eltwise_add_nc_vec_fxp(float16* host_inputs0,
                                float16* host_inputs1,
                                float16* host_results,
                                float* inputs0,
                                float* inputs1,
                                float* results,
                                int inputs_size);

void smv_eltwise_mul_nc_vec_fxp(float16* host_inputs0,
                                float16* host_inputs1,
                                float16* host_results,
                                float* inputs0,
                                float* inputs1,
                                float* results,
                                int inputs_size);

void smv_less_nc_vec_fxp(float16* host_inputs0,
                         float16* host_inputs1,
                         bool* host_results,
                         float* inputs0,
                         float* inputs1,
                         bool* results,
                         int inputs_size);

void smv_less_equal_nc_vec_fxp(float16* host_inputs0,
                               float16* host_inputs1,
                               bool* host_results,
                               float* inputs0,
                               float* inputs1,
                               bool* results,
                               int inputs_size);

void smv_greater_nc_vec_fxp(float16* host_inputs0,
                            float16* host_inputs1,
                            bool* host_results,
                            float* inputs0,
                            float* inputs1,
                            bool* results,
                            int inputs_size);

void smv_greater_equal_nc_vec_fxp(float16* host_inputs0,
                                  float16* host_inputs1,
                                  bool* host_results,
                                  float* inputs0,
                                  float* inputs1,
                                  bool* results,
                                  int inputs_size);
#ifdef __cplusplus
}
#endif

#endif
