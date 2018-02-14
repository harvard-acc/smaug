#ifndef _LAYERS_COMMON_H_
#define _LAYERS_COMMON_H_

#include "nnet_fwd.h"

// Convenience macro for profiling a kernel invocation.
#define INVOKE_KERNEL_PROF(req_code, lnum, kernel_ptr, args...)                \
    do {                                                                       \
        begin_profiling(STRING(kernel_ptr), lnum);                             \
        INVOKE_KERNEL(req_code, kernel_ptr, args);                             \
        end_profiling();                                                       \
    } while (0)

result_buf layer_dispatcher(float* activations,
                            float* weights,
                            layer_t* layers,
                            int layer_num,
                            float* result,
                            device_t* device,
                            sampling_param_t* sampling_param);
#endif
