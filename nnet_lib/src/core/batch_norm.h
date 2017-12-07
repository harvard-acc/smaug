#ifndef _BATCH_NORM_H_
#define _BATCH_NORM_H_

#include "nnet_fwd.h"

void batch_norm_fxp(float* inputs,
                    float* weights,
                    int input_size,
                    int batch_size,
                    float* result);

#endif
