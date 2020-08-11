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
 * Batch normalizes one input value.
 *
 * @param input Input activation.
 * @param mean Batch mean
 * @param recip_sqrt_var 1/sqrt(var + eps), which is precomputed to avoid
 * having to run a sqrt and division in the ASIC.
 * @param gamma Gamma parameter.
 * @param beta Beta parameter.
 */
ALWAYS_INLINE
v8fp_t batch_norm_simd_op(v8fp_t input,
                          v8fp_t mean,
                          v8fp_t recip_sqrt_var,
                          v8fp_t gamma,
                          v8fp_t beta) {
    v8fp_t scale = recip_sqrt_var * gamma;
    v8fp_t shift = input - mean;
    return shift * scale + beta;
}

/** \ingroup AladdinKernels
 *
 * SMV implementation of batch normalization following a fully-connected layer.
 *
 * In this case, we have one pair of gamma/beta weights per activation.
 */
void smv_batch_norm_post_fc_nc_vec_fxp(float16* host_inputs,
                                       float16* host_weights,
                                       float16* host_results,
                                       float* inputs,
                                       float* weights,
                                       float* results,
                                       int inputs_dims[2],
                                       int weights_acts,
                                       int inputs_pad,
                                       int inputs_start,
                                       int send_results,
                                       activation_type act_function,
                                       activation_param_t act_params) {
    int inputs_nums = inputs_dims[0];
    int inputs_acts = inputs_dims[1];
    int inputs_size = inputs_nums * (inputs_acts + inputs_pad);
    int weights_size = 4 * (weights_acts + inputs_pad);
    int results_size = inputs_size;
    int inputs_start_vec = inputs_start / VECTOR_SIZE;

    // Load inputs and weights if needed.
    if (inputs_start == 0)
        host_load_fp16(inputs, host_inputs, inputs_size, 0, 0);
    host_load_fp16(weights, host_weights, weights_size, 0, 0);

    VEC_ARRAY_2D(v8fp_t, _inputs, inputs, inputs_size + inputs_pad);
    VEC_ARRAY_2D(v8fp_t, _weights, weights, weights_acts + inputs_pad);
    VEC_ARRAY_2D(v8fp_t, _results, results, inputs_size + inputs_pad);

    bn_batch:
    for (int i = 0; i < inputs_nums; i++) {
        bn_input:
        for (int j = 0; j < weights_acts / VECTOR_SIZE; j++) {
            _results[i][j + inputs_start_vec] =
                    batch_norm_simd_op(_inputs[i][j + inputs_start_vec],
                                       _weights[0][j],
                                       _weights[1][j],
                                       _weights[2][j],
                                       _weights[3][j]);
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

/** \ingroup AladdinKernels
 *
 * SMV implementation of batch normalization following a convolutional/pooling
 * layer on NCHW data.
 *
 * After conv/pooling, we only have a gamma/beta per output feature map, not
 * per activation.
 */
void smv_batch_norm_post_conv_nchw_vec_fxp(float16* host_inputs,
                                           float16* host_weights,
                                           float16* host_results,
                                           float* inputs,
                                           float* weights,
                                           float* results,
                                           int inputs_dims[4],
                                           int weights_chans,
                                           int inputs_pad,
                                           int weights_pad,
                                           int weights_start,
                                           activation_type act_function,
                                           activation_param_t act_params) {
    int inputs_nums = inputs_dims[0];
    int inputs_chans = inputs_dims[1];
    int inputs_rows = inputs_dims[2];
    int inputs_cols = inputs_dims[3];
    int inputs_size = inputs_nums * inputs_chans * inputs_rows *
                      (inputs_cols + inputs_pad);
    int weights_size = 4 * (weights_chans + weights_pad);
    int results_size = inputs_size;
    int weights_start_vec = weights_start / VECTOR_SIZE;

    // Load inputs and weights if needed.
    host_load_fp16(inputs, host_inputs, inputs_size, 0, 0);
    if (weights_start == 0)
        host_load_fp16(weights, host_weights, weights_size, 0, 0);

    VEC_ARRAY_4D(v8fp_t,
                 _inputs,
                 inputs,
                 inputs_chans,
                 inputs_rows,
                 inputs_cols + inputs_pad);
    VEC_ARRAY_2D(v8fp_t, _weights, weights, weights_chans + weights_pad);
    VEC_ARRAY_4D(v8fp_t,
                 _results,
                 results,
                 inputs_chans,
                 inputs_rows,
                 inputs_cols + inputs_pad);

    bn_batch:
    for (int i = 0; i < inputs_nums; i++) {
        bn_chan:
        for (int h = 0; h < FRAC_CEIL(inputs_chans, VECTOR_SIZE); h++) {
            bn_chan_vec:
            for (int v = 0; v < VECTOR_SIZE; v++) {
                float mean = _weights[0][h + weights_start_vec][v];
                float recip_sqrt_var = _weights[1][h + weights_start_vec][v];
                float gamma = _weights[2][h + weights_start_vec][v];
                float beta = _weights[3][h + weights_start_vec][v];
                v8fp_t mean_vec = { mean, mean, mean, mean,
                                    mean, mean, mean, mean };
                v8fp_t recip_sqrt_var_vec = { recip_sqrt_var, recip_sqrt_var,
                                              recip_sqrt_var, recip_sqrt_var,
                                              recip_sqrt_var, recip_sqrt_var,
                                              recip_sqrt_var, recip_sqrt_var };
                v8fp_t gamma_vec = { gamma, gamma, gamma, gamma,
                                     gamma, gamma, gamma, gamma };
                v8fp_t beta_vec = { beta, beta, beta, beta,
                                    beta, beta, beta, beta };

                int ofmap = h * VECTOR_SIZE + v;
                bn_row:
                for (int r = 0; r < inputs_rows; r++) {
                    bn_col:
                    for (int c = 0; c < FRAC_CEIL(inputs_cols, VECTOR_SIZE);
                         c++) {
                        _results[i][ofmap][r][c] =
                                batch_norm_simd_op(_inputs[i][ofmap][r][c],
                                                   mean_vec,
                                                   recip_sqrt_var_vec,
                                                   gamma_vec,
                                                   beta_vec);
                    }
                }
            }
        }
    }
    if (act_function != NO_ACTIVATION) {
        activation_fun_vec(
                results, results, results_size, act_function, act_params);
    }
    // Store results to the host memory.
    host_store_fp16(results, host_results, results_size, 0, 0);
}

/** \ingroup AladdinKernels
 *
 * SMV implementation of batch normalization following a convolutional/pooling
 * layer on NHWC data.
 *
 * After conv/pooling, we only have a gamma/beta per output feature map, not
 * per activation.
 */
void smv_batch_norm_post_conv_nhwc_vec_fxp(float16* host_inputs,
                                           float16* host_weights,
                                           float16* host_results,
                                           float* inputs,
                                           float* weights,
                                           float* results,
                                           int inputs_dims[4],
                                           int weights_chans,
                                           int inputs_pad,
                                           int weights_pad,
                                           int weights_start,
                                           activation_type act_function,
                                           activation_param_t act_params,
                                           SamplingInfo* sampling) {
    int inputs_nums = inputs_dims[0];
    int inputs_rows = inputs_dims[1];
    int inputs_cols = inputs_dims[2];
    int inputs_chans = inputs_dims[3];
    int inputs_size = inputs_nums * inputs_rows * inputs_cols *
                      (inputs_chans + inputs_pad);
    int weights_size = 4 * (weights_chans + weights_pad);
    int results_size = inputs_size;
    int weights_start_vec = weights_start / VECTOR_SIZE;
    int inputs_chans_vec = FRAC_CEIL(inputs_chans, VECTOR_SIZE);

    // Load inputs and weights if needed.
    host_load_fp16(inputs, host_inputs, inputs_size, 0, 0);
    if (weights_start == 0)
        host_load_fp16(weights, host_weights, weights_size, 0, 0);

    VEC_ARRAY_4D(v8fp_t, _inputs, inputs, inputs_rows, inputs_cols,
                 inputs_chans + inputs_pad);
    VEC_ARRAY_2D(v8fp_t, _weights, weights, weights_chans + weights_pad);
    VEC_ARRAY_4D(v8fp_t, _results, results, inputs_rows, inputs_cols,
                 inputs_chans + inputs_pad);

    // We sample on the bn kernel only if the highest sampling level is
    // used.
    int batch_sample = inputs_nums;
    int chan_sample = inputs_chans_vec;
    int row_sample = inputs_rows;
    int col_sample = inputs_cols;
    int sample_num = sampling->num_sample_iterations;
    if (sampling->level >= VeryHigh) {
        batch_sample = min2(batch_sample, sample_num);
        chan_sample = min2(chan_sample, sample_num);
        row_sample = min2(row_sample, sample_num);
        col_sample = min2(col_sample, sample_num);
    }
    setSamplingFactor("bn_batch", inputs_nums * 1.0 / batch_sample);
    setSamplingFactor("bn_chan", inputs_chans_vec * 1.0 / chan_sample);
    setSamplingFactor("bn_row", inputs_rows * 1.0 / row_sample);
    setSamplingFactor("bn_col", inputs_cols * 1.0 / col_sample);

    bn_batch:
    for (int i = 0; i < batch_sample; i++) {
        bn_chan:
        for (int h = 0; h < chan_sample; h++) {
            v8fp_t mean = _weights[0][h + weights_start_vec];
            v8fp_t recip_sqrt_var = _weights[1][h + weights_start_vec];
            v8fp_t gamma = _weights[2][h + weights_start_vec];
            v8fp_t beta = _weights[3][h + weights_start_vec];
            bn_row:
            for (int r = 0; r < row_sample; r++) {
                bn_col:
                for (int c = 0; c < col_sample; c++) {
                    _results[i][r][c][h] =
                            batch_norm_simd_op(_inputs[i][r][c][h],
                                               mean,
                                               recip_sqrt_var,
                                               gamma,
                                               beta);
                }
            }
        }
    }
    if (act_function != NO_ACTIVATION) {
        activation_fun_vec(
                results, results, results_size, act_function, act_params);
    }
    // Store results to the host memory.
    host_store_fp16(results, host_results, results_size, 0, 0);
}

#ifdef __cplusplus
}  // extern "C"
#endif
