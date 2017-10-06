#ifndef _SMIV_CORE_H_
#define _SMIV_CORE_H_

void matrix_multiply_with_bias_smiv(float* a,
                                    float* b,
                                    int a_height,
                                    int b_height,
                                    int b_width,
                                    bool run_activation,
                                    float* result);

void reduction_smiv(float *a,
                    layer_t curr_layer,
                    int img,
                    int kern,
                    float *result);

void convolution2d_smiv(float* a,
                        float* kernels,
                        layer_t curr_layer,
                        float* result);

#endif
