#include "smaug/operators/common.h"
#include "smaug/operators/smv/kernels/load_store_fp16_data.h"
#include "smaug/operators/smv/kernels/activation_functions_simd.h"

#ifdef __cplusplus
extern "C" {
#endif

/** \ingroup AladdinKernels
 *
 * Top level function entry for all unary SMV activation functions.
 */
void smv_activation_fun_nc_vec_fxp(float16* host_inputs,
                                   float16* host_results,
                                   float* inputs,
                                   float* results,
                                   int inputs_size,
                                   activation_type function,
                                   activation_param_t params) {
    // Load inputs.
    host_load_fp16(inputs, host_inputs, inputs_size, 0, 0);
    activation_fun_vec(inputs, results, inputs_size, function, params);
    // Store results to the host memory.
    host_store_fp16(results, host_results, inputs_size, 0, 0);
}

/** \ingroup AladdinKernels
 *
 * Top level function for softmax.
 */
void smv_softmax_nc_vec_fxp(float16* host_inputs,
                            float16* host_results,
                            float* inputs,
                            float* results,
                            int input_num,
                            int input_size,
                            int input_pad) {
    // Load inputs.
    host_load_fp16(
            inputs, host_inputs, input_num * (input_size + input_pad), 0, 0);

    VEC_ARRAY_2D(v8fp_t, _inputs, inputs, input_size + input_pad);
    VEC_ARRAY_2D(v8fp_t, _results, results, input_size + input_pad);
    int input_vec_size = input_size / VECTOR_SIZE;

    softmax_batch:
    for (int i = 0; i < input_num; i++) {
        // Exponentiate.
        softmax_exp:
        for (int j = 0; j < input_vec_size; j++) {
            softmax_exp_vec:
            for (int k = 0; k < VECTOR_SIZE; k++)
                _results[i][j][k] = exp(_inputs[i][j][k]);
        }

        // Compute the normalization factor.
        float normaliz = 0.0;
        softmax_reduce:
        for (int j = 0; j < input_vec_size; j++) {
            softmax_reduce_vec:
            for (int k = 0; k < VECTOR_SIZE; k++)
                normaliz += _results[i][j][k];
        }

        // Precompute the division so that later we can just do a
        // multiplication.
        normaliz = 1.0 / (normaliz + 1e-6);  // epsilon for numerical stability.

        softmax_mul:
        for (int j = 0; j < input_vec_size; j++) {
            softmax_mul_vec:
            for (int k = 0; k < VECTOR_SIZE; k++)
                _results[i][j][k] *= normaliz;
        }
    }

    // Store results to the host memory.
    host_store_fp16(
            results, host_results, input_num * (input_size + input_pad), 0, 0);
}

#ifdef __cplusplus
}  // extern "C"
#endif
