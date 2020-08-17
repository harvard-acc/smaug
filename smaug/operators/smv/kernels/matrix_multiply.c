#include <assert.h>
#include <stdio.h>

#include "smaug/operators/common.h"
#include "smaug/operators/smv/kernels/params.h"
#include "smaug/operators/smv/kernels/load_store_fp16_data.h"
#include "smaug/operators/smv/kernels/activation_functions_simd.h"

#ifdef __cplusplus
extern "C" {
#endif

/** \ingroup AladdinKernels
 *
 * Matrix b after transposition:
 *
 * cols (originally rows) --->
 *
 * rows  [[---][---][---]]
 *  |    [[---][---][---]]
 *       [[---][---][---]]
 *  v    [[---][---][---]]
 *
 *  Each [---] represents an 8-wide vector. This inner product executes a 32-way
 *  MACC -- 4 such 8-wide vectors -- per PE, and 8 PEs, where each PE is
 *  assigned a row in in the transposed matrix. It continues across each row of
 *  b until the complete output pixel is finished (output stationary).
 *
 *  No biases are added.
 *
 * Args:
 * @param host_a Host buffer for a in NC.
 * @param host_b Host buffer for b in NC.
 * @param host_results Host results buffer in NC.
 * @param a Local buffer for a in NC.
 * @param b Local buffer for b in NC.
 * @param results Local results buffer in NC.
 * @param a_dims Dimensions of a.
 * @param b_dims Dimensions of b.
 * @param results_dims Dimensions of the results.
 * @param a_pad Align padding size on the channel dimension of a.
 * @param b_pad Align padding size on the channel dimension of b.
 * @param results_pad Align padding size on the channel dimension of the
 *        results.
 * @param a_start If a contains more activations than b, start from this one.
 *        Otherwise this should always be zero.
 * @param result_start If the results contain more neurons than the b, start
 *        writing results from this one. Otherwise this should always be zero.
 * @param accumulate If the original b tensor is tiled on activations, this
 *        should be set to true in order to avoid resetting the result buffer
 *        for knon-first b tiles.
 * @param read_inputs Load inputs from the host. Set to false if the input
 *        activations can be reused from the last invocation.
 * @param send_results Send the results to the host memory if this is true.
 * @param act_function Activation function the operator runs.
 * @param act_params Parameters for the activation function.
 * @param sampling Simulation samplng settings.
 */
void smv_matrix_multiply_transpose_nc_vec_fxp(float16* host_a,
                                              float16* host_b,
                                              float16* host_results,
                                              float* a,
                                              float* b,
                                              float* results,
                                              int a_dims[2],
                                              int b_dims[2],
                                              int results_dims[2],
                                              int a_pad,
                                              int b_pad,
                                              int results_pad,
                                              int a_start,
                                              int result_start,
                                              bool accumulate,
                                              bool read_inputs,
                                              bool send_results,
                                              activation_type act_function,
                                              activation_param_t act_params,
                                              SamplingInfo* sampling) {
    int a_width = a_dims[1];
    int a_height = a_dims[0];
    int b_width = b_dims[1];
    int b_height = b_dims[0];
    int results_width = results_dims[1];
    int results_height = results_dims[0];
    ASSERT((b_width + b_pad) % VECTOR_SIZE == 0 &&
           "Width of b must be a multiple of VECTOR_SIZE!");
    int a_width_vec = (a_width + a_pad) / VECTOR_SIZE;
    int b_width_vec = (b_width + b_pad) / VECTOR_SIZE;
    int a_size = a_height * (a_width + a_pad);
    int b_size = b_height * (b_width + b_pad);
    int results_size = results_height * (results_width + results_pad);

    v8fp_t zero = (v8fp_t){ 0, 0, 0, 0, 0, 0, 0, 0 };
    VEC_ARRAY_2D(v8fp_t, _a, a, a_width + a_pad);
    VEC_ARRAY_2D(v8fp_t, _b, b, b_width + b_pad);
    VEC_ARRAY_2D(v8fp_t, _results, results, results_width + results_pad);
    v8fp_t partial_sums;

    // Load a and b if needed.
    if (read_inputs)
        host_load_fp16(a, host_a, a_size, 0, 0);
    host_load_fp16(b, host_b, b_size, 0, 0);

    // We sample on the FC kernel only if the highest sampling level is used.
    int b_col_sample = b_width_vec;
    int b_col_total_iters = FRAC_CEIL(b_width_vec, NUM_MACC_INSTS);
    int b_col_sample_iters = b_col_total_iters;
    int sample_num = sampling->num_sample_iterations;
    if (sampling->level >= VeryHigh) {
        // Pipelined loops need at minimum 2 sampled iterations.
        b_col_sample_iters = min2(b_col_sample_iters, max2(2, sample_num));
        b_col_sample = b_col_sample_iters * NUM_MACC_INSTS;
    }
    setSamplingFactor("b_col", b_col_total_iters * 1.0 / b_col_sample_iters);

    a_act:
    for (int a_act = 0; a_act < a_height; a_act++) {
        b_row:
        for (int b_row = 0; b_row < b_height; b_row += NUM_PE_INSTS) {
            if (b_row % VECTOR_SIZE == 0) {
                if (accumulate) {
                    partial_sums = _results[a_act][(result_start + b_row) /
                                                   VECTOR_SIZE];
                } else {
                    partial_sums = zero;
                }
            }

            b_col:
            for (int b_col = 0; b_col < b_col_sample; b_col += NUM_MACC_INSTS) {
                // To work around an Aladdin dependence analysis bug where
                // InsertElement operations on vector types can be
                // serialized across unrolled loop iterations, we use a
                // normal scalar array here instead. Prior to committing the
                // data to the scratchpad, we'll copy this data back to a
                // vector register.
                float partial_sums_inner[VECTOR_SIZE] = {
                    0, 0, 0, 0, 0, 0, 0, 0
                };

                v8fp_t a_reg[NUM_MACC_INSTS];
                a_reg_load:
                for (int a_vec = 0; a_vec < NUM_MACC_INSTS; a_vec++) {
                    int a_col = a_start / VECTOR_SIZE + b_col + a_vec;
                    a_reg[a_vec] =
                            a_col >= a_width_vec ? zero : _a[a_act][a_col];
                }

                pe_insts:
                for (int pe_id = 0; pe_id < NUM_PE_INSTS; pe_id++) {
                    v8fp_t b_reg[NUM_MACC_INSTS];
                    b_reg_load:
                    for (int macc_idx = 0; macc_idx < NUM_MACC_INSTS;
                         macc_idx++) {
                        int pe_row = b_row + pe_id;
                        int this_b_col = b_col + macc_idx;
                        b_reg[macc_idx] =
                                (pe_row >= b_height ||
                                 this_b_col >= b_width_vec)
                                        ? zero
                                        : _b[pe_row][b_col + macc_idx];
                    }

                    v8fp_t product_reg[NUM_MACC_INSTS];
                    core_mul:
                    for (int macc_idx = 0; macc_idx < NUM_MACC_INSTS;
                         macc_idx++) {
                        product_reg[macc_idx] =
                                a_reg[macc_idx] * b_reg[macc_idx];
                    }

                    v8fp_t accum_vec_reg = zero;
                    reduce_1:
                    for (int macc_idx = 0; macc_idx < NUM_MACC_INSTS;
                         macc_idx++) {
                        accum_vec_reg += product_reg[macc_idx];
                    }

                    float accum_reg = 0;
                    reduce_2:
                    for (int vec_i = 0; vec_i < VECTOR_SIZE; vec_i++) {
                        accum_reg += accum_vec_reg[vec_i];
                    }
                    partial_sums_inner[pe_id] += accum_reg;
                }
                copy_psums:
                for (int i = 0; i < NUM_PE_INSTS; i++) {
                    partial_sums[i] += partial_sums_inner[i];
                }
            }

            int next_b_row = b_row + NUM_PE_INSTS;
            if (next_b_row % VECTOR_SIZE == 0 || next_b_row >= b_height) {
                _results[a_act][(result_start + b_row) / VECTOR_SIZE] =
                        partial_sums;
            }
        }
    }
    // Only run activation functions when the results are finished.
    if (act_function != NO_ACTIVATION && send_results) {
        activation_fun_vec(
                results, results, results_size, act_function, act_params);
    }
    // Store results to the host memory if needed.
    if (send_results)
        host_store_fp16(results, host_results, results_size, 0, 0);
}

#ifdef __cplusplus
}  // extern "C"
#endif
