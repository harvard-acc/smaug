#ifndef _SMIV_CORE_H_
#define _SMIV_CORE_H_

extern const unsigned VECTOR_SIZE;
extern const unsigned DATAPATH_WIDTH;
extern const unsigned SHIFT_REG_SIZE;
extern const unsigned MAX_BATCH;

void matrix_multiply_with_bias_smiv(float* a,
                                    float* b,
                                    int a_height,
                                    int b_height,
                                    int b_width,
                                    float* result);

void convolution2d_kernel_smiv(float* a,
                               float* kernels,
                               int img,
                               int kern,
                               layer_t curr_layer,
                               float* result);

#endif
