#ifndef _SMIV_CORE_H_
#define _SMIV_CORE_H_

void matrix_multiply_with_bias_smiv(float* a,
                                    float* b,
                                    int a_height,
                                    int b_height,
                                    int b_width,
                                    int a_pad,
                                    bool run_activation,
                                    bool do_bias,
                                    float* result);

void reduction_smiv(float* a, layer_t curr_layer, float* result);

void convolution3d_smiv(float* a,
                        float* kernels,
                        layer_t curr_layer,
                        int start_chan,
                        float* result);

void batch_norm_simd_fxp(float* inputs,
                         float* weights,
                         const layer_t* curr_layer,
                         int batch_size,
                         float* result);

void maxpooling_nhwc_smiv(float* inputs, layer_t curr_layer, float* results);

#endif
