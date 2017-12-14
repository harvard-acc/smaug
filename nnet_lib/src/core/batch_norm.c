#include "utility/utility.h"
#include "nnet_fwd.h"

#include "batch_norm.h"

// The weights are divided into four blocks.
enum {
  MEAN,
  VARIANCE,
  GAMMA,
  BETA
};

// 1/sqrt(var + eps) is precomputed to avoid having to run a sqrt and division
// in the ASIC.
ALWAYS_INLINE
inline float batch_norm_op(float input,
                           float mean,
                           float recip_sqrt_var,
                           float gamma,
                           float beta) {
    return ((input - mean) * recip_sqrt_var) * gamma + beta;
}

// For batch norm following a FC layer, we have one pair of gamma/beta weights
// per activation.
void batch_norm_post_fc_fxp(float* inputs,
                            float* weights,
                            layer_t* curr_layer,
                            int batch_size,
                            float* result) {
    int i, j;
    int input_size = curr_layer->inputs.rows * curr_layer->inputs.height *
                     (curr_layer->inputs.cols + curr_layer->inputs.align_pad);
    ARRAY_2D(float, _weights, weights, input_size);
    ARRAY_2D(float, _inputs, inputs, input_size);
    ARRAY_2D(float, _result, result, input_size);

    bn_batch:
    for (i = 0; i < batch_size; i++) {
        bn_input:
        for (j = 0; j < input_size; j++) {
            float mean = _weights[MEAN][j];
            float recip_sqrt_var = _weights[VARIANCE][j];
            float gamma = _weights[GAMMA][j];
            float beta = _weights[BETA][j];
            _result[i][j] = batch_norm_op(
                    _inputs[i][j], mean, recip_sqrt_var, gamma, beta);
        }
    }
}

// For batch norm following a convolutional/pooling layer, we only have a
// gamma/beta per output feature map, not per activation.
void batch_norm_post_conv_fxp(float* inputs,
                              float* weights,
                              layer_t* curr_layer,
                              int batch_size,
                              float* result) {
    const int num_chans = curr_layer->inputs.height;
    const int weight_align_pad = curr_layer->weights.align_pad;
    const int input_rows = curr_layer->inputs.rows;
    const int input_cols = curr_layer->inputs.cols;
    const int input_align_pad = curr_layer->inputs.align_pad;
    const int output_rows = curr_layer->outputs.rows;
    const int output_cols = curr_layer->outputs.cols;
    const int output_align_pad = curr_layer->outputs.align_pad;

    ARRAY_2D(float, _weights, weights, num_chans + weight_align_pad);
    ARRAY_4D(float,
             _inputs,
             inputs,
             num_chans,
             input_rows,
             input_cols + input_align_pad);
    ARRAY_4D(float,
             _result,
             result,
             num_chans,
             output_rows,
             output_cols + output_align_pad);

    bn_batch:
    for (int i = 0; i < batch_size; i++) {
        bn_chan:
        for (int h = 0; h < num_chans; h++) {
            float mean = _weights[MEAN][h];
            float recip_sqrt_var = _weights[VARIANCE][h];
            float gamma = _weights[GAMMA][h];
            float beta = _weights[BETA][h];

            bn_row:
            for (int r = 0; r < input_rows; r++) {
                bn_col:
                for (int c = 0; c < input_cols + input_align_pad; c++) {
                    _result[i][h][r][c] = batch_norm_op(_inputs[i][h][r][c],
                                                        mean,
                                                        recip_sqrt_var,
                                                        gamma,
                                                        beta);
                }
            }
        }
    }
}

// Perform batch normalization on the data in @input.
void batch_norm_fxp(float* inputs,
                    float* weights,
                    layer_t* curr_layer,
                    int batch_size,
                    float* result) {

    if (curr_layer->inputs.height == 1) {
        batch_norm_post_fc_fxp(inputs, weights, curr_layer, batch_size, result);
    } else {
        batch_norm_post_conv_fxp(
                inputs, weights, curr_layer, batch_size, result);
    }
}
