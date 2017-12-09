#ifndef _EIGEN_CONVOLUTION_H_
#define _EIGEN_CONVOLUTION_H_

#include "nnet_fwd.h"

namespace nnet_eigen {

void convolution3d(float* activations,
                   float* kernels,
                   layer_t* curr_layer,
                   float* result);

}  // namespace nnet_eigen

#endif
