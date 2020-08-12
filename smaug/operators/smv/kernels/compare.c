#include "smaug/operators/common.h"
#include "smaug/operators/smv/kernels/params.h"
#include "smaug/operators/smv/kernels/load_store_fp16_data.h"

#ifdef __cplusplus
extern "C" {
#endif

ALWAYS_INLINE
static inline v8bl_t convert_to_bool(v8sfx_t a) {
    return (v8bl_t){ (bool)a[0], (bool)a[1], (bool)a[2], (bool)a[3],
                     (bool)a[4], (bool)a[5], (bool)a[6], (bool)a[7] };
}

/** \ingroup AladdinKernels
 *
 * SMVe implementation of elementwise less-than.
 */
void smv_less_nc_vec_fxp(float16* host_inputs0,
                         float16* host_inputs1,
                         bool* host_results,
                         float* inputs0,
                         float* inputs1,
                         bool* results,
                         int inputs_size) {
    // Load inputs.
    host_load_fp16(inputs0, host_inputs0, inputs_size, 0, 0);
    host_load_fp16(inputs1, host_inputs1, inputs_size, 0, 0);

    VEC_ARRAY_1D(v8fp_t, _inputs0, inputs0);
    VEC_ARRAY_1D(v8fp_t, _inputs1, inputs1);
    VEC_ARRAY_1D(v8bl_t, _results, results);

    less_loop:
    for (int i = 0; i < inputs_size / VECTOR_SIZE; i++) {
        v8sfx_t result = _inputs0[i] < _inputs1[i];
        _results[i] = convert_to_bool(result);
    }

    // Store results to the host memory.
    dmaStore(host_results, results, inputs_size * sizeof(bool));
}

/** \ingroup AladdinKernels
 *
 * SMVe implementation of elementwise less-than-or-equal-to.
 */
void smv_less_equal_nc_vec_fxp(float16* host_inputs0,
                               float16* host_inputs1,
                               bool* host_results,
                               float* inputs0,
                               float* inputs1,
                               bool* results,
                               int inputs_size) {
    // Load inputs.
    host_load_fp16(inputs0, host_inputs0, inputs_size, 0, 0);
    host_load_fp16(inputs1, host_inputs1, inputs_size, 0, 0);

    VEC_ARRAY_1D(v8fp_t, _inputs0, inputs0);
    VEC_ARRAY_1D(v8fp_t, _inputs1, inputs1);
    VEC_ARRAY_1D(v8bl_t, _results, results);

    less_equal_loop:
    for (int i = 0; i < inputs_size / VECTOR_SIZE; i++) {
        v8sfx_t result = _inputs0[i] <= _inputs1[i];
        _results[i] = convert_to_bool(result);
    }

    // Store results to the host memory.
    dmaStore(host_results, results, inputs_size * sizeof(bool));
}

/** \ingroup AladdinKernels
 *
 * SMVe implementation of elementwise greater-than.
 */
void smv_greater_nc_vec_fxp(float16* host_inputs0,
                            float16* host_inputs1,
                            bool* host_results,
                            float* inputs0,
                            float* inputs1,
                            bool* results,
                            int inputs_size) {
    // Load inputs.
    host_load_fp16(inputs0, host_inputs0, inputs_size, 0, 0);
    host_load_fp16(inputs1, host_inputs1, inputs_size, 0, 0);

    VEC_ARRAY_1D(v8fp_t, _inputs0, inputs0);
    VEC_ARRAY_1D(v8fp_t, _inputs1, inputs1);
    VEC_ARRAY_1D(v8bl_t, _results, results);

    greater_loop:
    for (int i = 0; i < inputs_size / VECTOR_SIZE; i++) {
        v8sfx_t result = _inputs0[i] > _inputs1[i];
        _results[i] = convert_to_bool(result);
    }

    // Store results to the host memory.
    dmaStore(host_results, results, inputs_size * sizeof(bool));
}

/** \ingroup AladdinKernels
 *
 * SMVe implementation of elementwise greater-than-or-equal-to.
 */
void smv_greater_equal_nc_vec_fxp(float16* host_inputs0,
                                  float16* host_inputs1,
                                  bool* host_results,
                                  float* inputs0,
                                  float* inputs1,
                                  bool* results,
                                  int inputs_size) {
    // Load inputs.
    host_load_fp16(inputs0, host_inputs0, inputs_size, 0, 0);
    host_load_fp16(inputs1, host_inputs1, inputs_size, 0, 0);

    VEC_ARRAY_1D(v8fp_t, _inputs0, inputs0);
    VEC_ARRAY_1D(v8fp_t, _inputs1, inputs1);
    VEC_ARRAY_1D(v8bl_t, _results, results);

    greater_equal_loop:
    for (int i = 0; i < inputs_size / VECTOR_SIZE; i++) {
        v8sfx_t result = _inputs0[i] >= _inputs1[i];
        _results[i] = convert_to_bool(result);
    }

    // Store results to the host memory.
    dmaStore(host_results, results, inputs_size * sizeof(bool));
}

#ifdef __cplusplus
}  // extern "C"
#endif
