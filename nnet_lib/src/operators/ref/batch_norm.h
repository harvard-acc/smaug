#ifndef _BATCH_NORM_H_
#define _BATCH_NORM_H_

#include "core/nnet_fwd_defs.h"

void batch_norm_fxp(float* inputs,
                    float* weights,
                    const layer_t* curr_layer,
                    int batch_size,
                    float* result);

#endif
