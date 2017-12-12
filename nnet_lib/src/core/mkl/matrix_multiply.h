#ifndef _MKL_MATRIX_MULTIPLY_H_
#define _MKL_MATRIX_MULTIPLY_H_

#include "arch/nnet_mkl.h"

namespace nnet_mkl {

void matrix_multiply_with_bias(float* inputs,
                               float* weights,
                               layer_t* curr_layer,
                               float* results,
                               device_t* device);

}  // namespace nnet_mkl

#endif

