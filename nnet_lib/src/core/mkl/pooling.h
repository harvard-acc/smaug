#ifndef _MKL_POOLING_H_
#define _MKL_POOLING_H_

#include "core/nnet_fwd_defs.h"
#include "arch/nnet_mkl.h"

namespace nnet_mkl {

void max_pooling_3d(float* inputs,
                    layer_t* curr_layer,
                    float* result,
                    device_t* device);

}  // namespace nnet_mkl

#endif
