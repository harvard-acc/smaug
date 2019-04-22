#ifndef _OPERATORS_SMV_KERNELS_H_
#define _OPERATORS_SMV_KERNELS_H_

#ifdef __cplusplus
extern "C" {
#endif

void smv_conv3d_f32_nhwc_vec_fxp(float16* host_inputs,
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
                                 bool send_results);

void smv_matrix_multiply_transpose_f32_nc_vec_fxp(float16* host_a,
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
                                                  bool send_results);

#ifdef __cplusplus
}
#endif

#endif
