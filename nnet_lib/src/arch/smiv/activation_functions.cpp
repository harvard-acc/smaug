#include "core/nnet_fwd_defs.h"
#include "core/ref/activation_functions.h"
#include "utility/profiling.h"
#include "utility/utility.h"

#ifdef __cplusplus
#include "core/mkl/activation_functions.h"
#endif

float* smiv_activation_function_impl(float* activations,
                                     layer_t* layer,
                                     float* results,
                                     device_t* device) {
    // MKL's activation functions are very poor performing, possibly due to it
    // not being vectorized within the constraints of gem5. So use the
    // reference implementation for activation functions for now.
#ifdef SMIV_USE_MKL_ACTIVATION_FUNCTION_IMPL
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
    activation_fun(activations,
                   NUM_TEST_CASES,
                   output_size,
                   layer->activation);
    end_profiling();
    return activations;
#endif
}
