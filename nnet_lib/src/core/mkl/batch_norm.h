#ifndef _MKL_BATCH_NORM_H_
#define _MKL_BATCH_NORM_H_

#include "mkldnn.hpp"

#include "arch/nnet_mkl.h"
#include "core/nnet_fwd_defs.h"

namespace nnet_mkl {

void batch_norm(float* inputs,
                float* weights,
                layer_t* curr_layer,
                int batch_size,
                float* results,
                device_t* device);
}

#endif
