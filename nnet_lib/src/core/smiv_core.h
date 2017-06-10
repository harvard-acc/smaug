#ifndef _SMIV_CORE_H_
#define _SMIV_CORE_H_

#define VECTOR_SIZE 8
#define DATAPATH_WIDTH 4
#define SHIFT_REG_SIZE 16
#define MAX_BATCH 8

void matrix_multiply_with_bias_smiv(float* a,
                                    float* b,
                                    int a_height,
                                    int b_height,
                                    int b_width,
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
