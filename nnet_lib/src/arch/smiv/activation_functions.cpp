#include "core/nnet_fwd_defs.h"
#include "core/ref/activation_functions.h"
#include "utility/profiling.h"
#include "utility/utility.h"

#ifdef __cplusplus
#include "core/mkl/activation_functions.h"
#endif

result_buf smiv_activation_function(float* activations,
                                    layer_t* layer,
                                    float* results,
                                    device_t* device) {
    // MKL is giving unexpectedly worse performance than our reference
    // implementation. So use the reference implementation for activation
    // functions for now.
#if 0
    begin_ignored_profiling(layer->num);
    nnet_mkl::activation_fun(
            activations, NUM_TEST_CASES, layer, results, device);
    end_profiling();
    nnet_mkl::MklSession* session = nnet_mkl::get_session(device);
    session->run_and_clear();
    return results;
#else
    int output_size = get_dims_size(&layer->outputs);
    begin_profiling(ACTIVATION_TYPE_STR(layer->activation), layer->num);
    activation_fun(activations, NUM_TEST_CASES, output_size, layer->activation);
    end_profiling();
    return activations;
#endif
}

