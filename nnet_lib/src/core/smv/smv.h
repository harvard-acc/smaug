#ifndef _CORE_SMV_H_
#define _CORE_SMV_H_

#include "core/nnet_fwd_defs.h"

void convolution3d_smv(float* a,
                       float* kernels,
                       layer_t curr_layer,
                       int start_chan,
                       float* result);

#endif
