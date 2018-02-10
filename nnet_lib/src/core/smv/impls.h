#ifndef _SMV_IMPL_H_
#define _SMV_IMPL_H_

#include "core/nnet_fwd_defs.h"
#include "core/smiv/params.h"

void convolution3d_smv_nhwc_fxp(float* a,
                                float* kernels,
                                layer_t curr_layer,
                                int start_chan,
                                float* result);

#endif
