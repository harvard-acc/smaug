#ifndef _OPERATORS_SMV_KERNELS_H_
#define _OPERATORS_SMV_KERNELS_H_

#ifdef __cplusplus
extern "C" {
#endif

void smv_conv3d_f32_nhwc_vec_fxp(float* inputs,
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
                                 int ofmap_start,
                                 int ifmap_start,
                                 bool accumulate);

#ifdef __cplusplus
}
#endif

#endif
