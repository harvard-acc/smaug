#ifndef _EIGEN_FLATTEN_H_
#define _EIGEN_FLATTEN_H_

#include "core/nnet_fwd_defs.h"

namespace nnet_eigen {

void flatten_input(float* activations,
                   layer_t* curr_layer,
                   int num_images,
                   float* result);

}  // namespace nnet_eigen

#endif
