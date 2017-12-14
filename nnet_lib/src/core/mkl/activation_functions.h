#ifndef _MKL_ACTIVATION_FUNCTIONS_H_
#define _MKL_ACTIVATION_FUNCTIONS_H_

#include "mkldnn.hpp"

#include "arch/nnet_mkl.h"

namespace nnet_mkl {

void sigmoid(float* activations, int size, mkldnn::engine& cpu, float* results);
void relu(float* activations, int size, mkldnn::engine& cpu, float* results);
void activation_fun(float* activations,
                    int size,
                    activation_type function,
                    float* results,
                    device_t* device);
}

#endif
