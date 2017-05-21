#ifndef _MATRIX_MULTIPLY_
#define _MATRIX_MULTIPLY_

void matrix_multiply(float* a,
                     float* b,
                     int a_height,
                     int a_width_b_height,
                     int b_width,
                     float* result_goes_here,
                     float* result_temp);

void matrix_multiply_with_bias(float* a,
                               float* b,
                               int a_height,
                               int b_height,
                               int b_width,
                               float* result);

void matrix_multiply_with_bias_transpose(float* a,
                                         float* b,
                                         int a_height,
                                         int b_height,
                                         int b_width,
                                         float* result);

void matrix_multiply_with_bias_and_copy(float* a,
                                        float* b,
                                        int a_height,
                                        int b_height,
                                        int b_width,
                                        float* result_goes_here,
                                        float* result_temp);

void matrix_multiply_with_bias_smiv(float* a,
                                    float* b,
                                    int a_height,
                                    int b_height,
                                    int b_width,
                                    float* result);
#endif
