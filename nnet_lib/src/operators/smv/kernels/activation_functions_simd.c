#include "operators/common.h"
#include "load_store_fp16_data.h"
#include "activation_functions_simd.h"

#ifdef __cplusplus
extern "C" {
#endif

// Top level function entry for activation functions.
void smv_activation_fun_nc_vec_fxp(float16* host_inputs,
                                   float16* host_results,
                                   float* inputs,
                                   float* results,
                                   int inputs_size,
                                   activation_type function,
                                   activation_param_t params) {
    // Load inputs.
    dma_load_fp16(inputs, host_inputs, inputs_size, 0, 0);

    VEC_ARRAY_1D(v8fp_t, _inputs, inputs);
    VEC_ARRAY_1D(v8fp_t, _results, results);
    activation_fun_vec(_inputs, _results, inputs_size, function, params);

    // Store results to the host memory.
    dma_store_fp16(results, host_results, inputs_size, 0, 0);
}

#ifdef __cplusplus
}  // extern "C"
#endif
