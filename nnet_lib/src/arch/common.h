#ifndef _LAYERS_COMMON_H_
#define _LAYERS_COMMON_H_

#include "nnet_fwd.h"

result_buf layer_dispatcher(float* activations,
                            float* weights,
                            layer_t* layers,
                            int layer_num,
                            float* result,
                            device_t* device,
                            sampling_param_t* sampling_param);
#endif
