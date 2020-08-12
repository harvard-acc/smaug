#include "smaug/operators/common.h"
#include "smaug/operators/smv/kernels/params.h"
#include "smaug/operators/smv/kernels/load_store_fp16_data.h"

#ifdef __cplusplus
extern "C" {
#endif

/** \ingroup AladdinKernels
 *
 * SMV implementation of elementwise addition.
 */
void smv_eltwise_add_nc_vec_fxp(float16* host_inputs0,
                                float16* host_inputs1,
                                float16* host_results,
                                float* inputs0,
                                float* inputs1,
                                float* results,
                                int inputs_size) {
    // Load inputs.
    host_load_fp16(inputs0, host_inputs0, inputs_size, 0, 0);
    host_load_fp16(inputs1, host_inputs1, inputs_size, 0, 0);

    VEC_ARRAY_1D(v8fp_t, _inputs0, inputs0);
    VEC_ARRAY_1D(v8fp_t, _inputs1, inputs1);
    VEC_ARRAY_1D(v8fp_t, _results, results);

    eltwise_add_loop:
    for (int i = 0; i < inputs_size / VECTOR_SIZE; i++) {
        _results[i] = _inputs0[i] + _inputs1[i];
    }

    // Store results to the host memory.
    host_store_fp16(results, host_results, inputs_size, 0, 0);
}

#ifdef __cplusplus
}  // extern "C"
#endif
