#ifndef _EIGEN_MATRIX_MULTIPLY_H_
#define _EIGEN_MATRIX_MULTIPLY_H_

#ifdef EIGEN_ARCH_IMPL

namespace nnet_eigen {

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

};  // namespace nnet_eigen

#endif

#endif
