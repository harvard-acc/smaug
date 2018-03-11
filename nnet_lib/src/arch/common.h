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

// Convenience macro to sample a profiled kernel invocation.
#define INVOKE_KERNEL_SAMPLED(                                                 \
        req_code, lnum, sampled_iters, kernel_ptr, args...)                    \
    do {                                                                       \
        begin_profiling(STRING(kernel_ptr), lnum);                             \
        if ((sampled_iters) > 1)                                               \
            set_profiling_type_sampled(1, (sampled_iters));                    \
        INVOKE_KERNEL((req_code), (kernel_ptr), args);                         \
        end_profiling();                                                       \
    } while (0)

result_buf layer_dispatcher(data_list* activations,
                            data_list* weights,
                            layer_t* layers,
                            int layer_num,
                            data_list* result,
                            device_t* device,
                            sampling_param_t* sampling_param);
#endif
