#include "core/smiv/activation_functions_simd.h"
#include "core/smiv/params.h"
#include "core/nnet_fwd_defs.h"
#include "utility/utility.h"

// 1/sqrt(var + eps) is precomputed to avoid having to run a sqrt and division
// in the ASIC.
ALWAYS_INLINE
v8fp_t batch_norm_simd_op(v8fp_t input,
                          v8fp_t mean,
                          v8fp_t recip_sqrt_var,
                          v8fp_t gamma,
                          v8fp_t beta,
                          activation_type activation_func) {
    v8fp_t scale = recip_sqrt_var * gamma;
    v8fp_t shift = input - mean;
    v8fp_t result = shift * scale + beta;
    return activation_fun_simd_fxp(result, activation_func);
}

void batch_norm_post_fc_simd_fxp(float* inputs,
                                 float* weights,
                                 const layer_t* curr_layer,
                                 int batch_size,
                                 float* results) {
    int i, j;
    int input_size = curr_layer->inputs.rows * curr_layer->inputs.height *
                     (curr_layer->inputs.cols + curr_layer->inputs.align_pad);
    activation_type act = curr_layer->activation;
    VEC_ARRAY_2D(v8fp_t, _inputs, inputs, input_size);
    VEC_ARRAY_2D(v8fp_t, _weights, weights, input_size);
    VEC_ARRAY_2D(v8fp_t, _results, results, input_size);

    bn_batch:
    for (i = 0; i < batch_size; i++) {
        bn_input:
        for (j = 0; j < input_size / VECTOR_SIZE; j++) {
            v8fp_t mean = _weights[MeanIndex][j];
            v8fp_t recip_sqrt_var = _weights[VarianceIndex][j];
            v8fp_t gamma = _weights[GammaIndex][j];
            v8fp_t beta = _weights[BetaIndex][j];
            _results[i][j] = batch_norm_simd_op(
                    _inputs[i][j], mean, recip_sqrt_var, gamma, beta, act);
        }
    }
}

// For batch norm following a convolutional/pooling layer, we only have a
// gamma/beta per output feature map, not per activation.
void batch_norm_post_conv_simd_fxp(float* inputs,
                                   float* weights,
                                   const layer_t* curr_layer,
                                   int batch_size,
                                   float* results,
                                   int weight_col_start) {
    const int num_chans = curr_layer->inputs.height;
    const int weight_cols = curr_layer->weights.cols;
    const int weight_align_pad = curr_layer->weights.align_pad;
    const int input_rows = curr_layer->inputs.rows;
    const int input_cols = curr_layer->inputs.cols;
    const int input_align_pad = curr_layer->inputs.align_pad;
    const int output_rows = curr_layer->outputs.rows;
    const int output_cols = curr_layer->outputs.cols;
    const int output_align_pad = curr_layer->outputs.align_pad;
    const int input_cols_vec = FRAC_CEIL(input_cols, VECTOR_SIZE);
    activation_type act = curr_layer->activation;

    VEC_ARRAY_2D(v8fp_t, _weights, weights, weight_cols + weight_align_pad);
    VEC_ARRAY_4D(v8fp_t,
                 _inputs,
                 inputs,
                 num_chans,
                 input_rows,
                 input_cols + input_align_pad);
    VEC_ARRAY_4D(v8fp_t,
                 _results,
                 results,
                 num_chans,
                 output_rows,
                 output_cols + output_align_pad);

    bn_batch:
    for (int i = 0; i < batch_size; i++) {
        bn_chan:
        for (int h = 0; h < FRAC_CEIL(num_chans, VECTOR_SIZE); h++) {
            v8fp_t mean_vec =
                    _weights[MeanIndex][h + weight_col_start / VECTOR_SIZE];
            v8fp_t recip_sqrt_var_vec =
                    _weights[VarianceIndex][h + weight_col_start / VECTOR_SIZE];
            v8fp_t gamma_vec =
                    _weights[GammaIndex][h + weight_col_start / VECTOR_SIZE];
            v8fp_t beta_vec =
                    _weights[BetaIndex][h + weight_col_start / VECTOR_SIZE];

            bn_chan_vec:
            for (int v = 0; v < VECTOR_SIZE; v++) {
                float mean = mean_vec[v];
                float recip_sqrt_var = recip_sqrt_var_vec[v];
                float gamma = gamma_vec[v];
                float beta = beta_vec[v];
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

                bn_row:
                for (int r = 0; r < input_rows; r++) {
                    bn_col:
                    for (int c = 0; c < input_cols_vec; c++) {
                        int ofmap = h * VECTOR_SIZE + v;
                        _results[i][ofmap][r][c] =
                                batch_norm_simd_op(_inputs[i][ofmap][r][c],
                                                   mean_vec,
                                                   recip_sqrt_var_vec,
                                                   gamma_vec,
                                                   beta_vec,
                                                   act);
                    }
                }
            }
        }
    }
}

// Perform batch normalization on the data in @input.
void batch_norm_simd_fxp(float* inputs,
                         float* weights,
                         const layer_t* curr_layer,
                         int batch_size,
                         float* results,
                         int weight_col_start) {
    // TODO: This will break if the output of a conv/pool/input layer has
    // height 1, since it will be interpreted as the output of an FC layer.
    // We really need to fix this problem of using layer_t to glob the
    // configurations of EVERY layer time.
    if (curr_layer->inputs.height == 1) {
        batch_norm_post_fc_simd_fxp(
                inputs, weights, curr_layer, batch_size, results);
    } else {
        batch_norm_post_conv_simd_fxp(inputs,
                                      weights,
                                      curr_layer,
                                      batch_size,
                                      results,
                                      weight_col_start);
    }
}
