#ifndef _MKL_CONVOLUTION_H_
#define _MKL_CONVOLUTION_H_

#include "arch/nnet_mkl.h"
#include "utility/utility.h"

namespace nnet_mkl {

void convolution3d(float* inputs,
                   float* weights,
                   layer_t* curr_layer,
                   float* results,
                   device_t* device);

}  // namespace nnet_mkl

#endif
