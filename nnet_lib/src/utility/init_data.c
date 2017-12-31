#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#include "utility/utility.h"
#include "nnet_fwd.h"

#include "init_data.h"

inline float gen_uniform() {
    return randfloat();
}

// Returns an approximately normally distributed random value, using the
// Box-Muller method.
float gen_gaussian() {
    static bool return_saved = false;
    static float saved = 0;

    if (return_saved) {
        return_saved = false;
        return saved;
    } else {
        float u = gen_uniform();
        float v = gen_uniform();
        float scale = sqrt(-2 * log(u));
        float x = scale * cos(2 * 3.1415926535 * v);
        float y = scale * sin(2 * 3.1415926535 * v);
        saved = y;
        return_saved = true;
        return x;
    }
}

float get_rand_weight(data_init_mode mode, int depth) {
    if (mode == RANDOM) {
        // Question: does nan output take longer in simulation?
        return conv_float2fixed(gen_gaussian());
    } else {
        // Give each depth slice a different weight so we don't get all the
        // same value in the output.
        return (depth + 1) * 0.1;
    }
}

void init_fc_weights(float* weights,
                     int w_height,
                     int w_rows,
                     int w_cols,
                     int w_pad,
                     data_init_mode mode,
                     bool transpose) {
    int w_tot_cols = w_cols + w_pad;
    float val = 0;
    // Always store the biases after all the weights, regardless of whether
    // weights are transposed or not.
    int w_rows_minus_1 = w_rows - 1;
    for (int h = 0; h < w_height; h++) {
        // First store the weights.
        for (int i = 0; i < w_rows_minus_1; i++) {
            for (int j = 0; j < w_tot_cols; j++) {
                if (j < w_cols) {
                    val = get_rand_weight(mode, 0);
                } else {  // extra zero padding.
                    val = 0;
                }
                if (transpose)
                    weights[sub3ind(h, j, i, w_tot_cols, w_rows_minus_1)] = val;
                else
                    weights[sub3ind(h, i, j, w_rows_minus_1, w_tot_cols)] = val;
            }
        }
        // Store the biases.
        for (int j = 0; j < w_tot_cols; j++) {
            if (j < w_cols) {
                val = get_rand_weight(mode, 0);
            } else {
                val = 0;
            }
            weights[sub3ind(h, w_rows - 1, j, w_rows, w_tot_cols)] = val;
        }
    }
}

void init_conv_weights(float* weights,
                       int w_depth,
                       int w_height,
                       int w_rows,
                       int w_cols,
                       int w_pad,
                       data_init_mode mode,
                       bool transpose) {
    int w_tot_cols = w_cols + w_pad;
    float val = 0;
    for (int d = 0; d < w_depth; d++) {
        for (int h = 0; h < w_height; h++) {
            for (int i = 0; i < w_rows; i++) {
                for (int j = 0; j < w_tot_cols; j++) {
                    if (j < w_cols) {
                        val= get_rand_weight(mode, d);
                    } else {  // extra zero padding.
                        val = 0;
                    }
                    weights[sub4ind(d, h, i, j, w_height, w_rows, w_tot_cols)] =
                            val;
                }
            }
        }
    }
}

void init_bn_weights(float* weights,
                     int w_height,
                     int w_rows,
                     int w_cols,
                     int w_pad,
                     data_init_mode mode,
                     bool precompute_variance) {
    static const float kEpsilon = 1e-5;

    int w_tot_cols = w_cols + w_pad;
    float val = 0;
    for (int h = 0; h < w_height; h++) {
        for (int i = 0; i < w_rows; i++) {
            // BN parameters are stored in blocks of w_rows * w_tot_cols.
            // The block order is:
            //   1. mean
            //   2. variance
            //   3. gamma
            //   4. beta
            for (int j = 0; j < w_tot_cols; j++) {
                if (j < w_cols) {
                    val = get_rand_weight(mode, 0);
                } else {  // extra zero padding.
                    val = 0;
                }
                bool is_variance_block = (i / (w_rows / 4)) == 1;
                if (is_variance_block) {
                    // Variance cannot be negative.
                    val = val < 0 ? -val : val;
                    if (precompute_variance) {
                        // Precompute 1/sqrt(var + eps) if ARCH is not MKLDNN
                        // (in MKLDNN, we can't do this trick yet).
                        val = 1.0 / (sqrt(val + kEpsilon));
                    }
                }

                weights[sub3ind(h, i, j, w_rows, w_tot_cols)] = val;
            }
        }
    }
}

void init_weights(float* weights,
                  layer_t* layers,
                  int num_layers,
                  data_init_mode mode,
                  bool transpose) {
    int l;
    int w_rows, w_cols, w_height, w_depth, w_offset, w_pad;

    assert(mode == RANDOM || mode == FIXED);
    w_offset = 0;
    printf("Initializing weights randomly\n");

    for (l = 0; l < num_layers; l++) {
        get_weights_dims_layer(
                layers, l, &w_rows, &w_cols, &w_height, &w_depth, &w_pad);
        int w_tot_cols = w_cols + w_pad;
        switch (layers[l].type) {
            case FC:
                init_fc_weights(weights + w_offset, w_height, w_rows, w_cols,
                                w_pad, mode, transpose);
                break;
            case CONV_STANDARD:
            case CONV_DEPTHWISE:
            case CONV_POINTWISE:
                init_conv_weights(weights + w_offset, w_depth, w_height, w_rows,
                                  w_cols, w_pad, mode, transpose);
                break;
            case BATCH_NORM:
                init_bn_weights(weights + w_offset, w_height, w_rows, w_cols,
                                w_pad, mode, PRECOMPUTE_BN_VARIANCE);
                break;
            default:
                continue;
        }
        w_offset += w_rows * w_tot_cols * w_height * w_depth;
    }
    // NOTE: FOR SIGMOID ACTIVATION FUNCTION, WEIGHTS SHOULD BE BIG
    // Otherwise everything just becomes ~0.5 after sigmoid, and results are
    // boring
}

void init_data(float* data,
               network_t* network,
               size_t num_test_cases,
               data_init_mode mode) {
    unsigned i;
    int j, k, l;
    int input_rows, input_cols, input_height, input_align_pad;
    int input_dim;

    input_rows = network->layers[0].inputs.rows;
    input_cols = network->layers[0].inputs.cols;
    input_height = network->layers[0].inputs.height;
    input_align_pad = network->layers[0].inputs.align_pad;
    input_dim = input_rows * input_cols * input_height;

    ARRAY_4D(float, _data, data, input_height, input_rows,
             input_cols + input_align_pad);
    assert(mode == RANDOM || mode == FIXED);
    printf("Initializing data randomly\n");
    // Generate random input data, size num_test_cases by num_units[0]
    // (input dimensionality)
    for (i = 0; i < num_test_cases; i++) {
        int offset = 0;
        for (j = 0; j < input_height; j++) {
            for (k = 0; k < input_rows; k++) {
                for (l = 0; l < input_cols; l++) {
                    if (mode == RANDOM) {
                        _data[i][j][k][l] = conv_float2fixed(gen_gaussian());
                    } else {
                        // Make each input image distinguishable.
                        _data[i][j][k][l] = 1.0 * i + (float)offset / input_dim;
                        offset++;
                    }
                }
                for (l = input_cols; l < input_cols + input_align_pad; l++) {
                    _data[i][j][k][l] = 0;
                }
            }
        }
    }
    PRINT_MSG("Input activations:\n");
    PRINT_DEBUG4D(data, input_rows, input_cols + input_align_pad, input_height);
}

void init_labels(int* labels, size_t label_size, data_init_mode mode) {
    unsigned i;
    assert(mode == RANDOM || mode == FIXED);
    printf("Initializing labels randomly\n");
    for (i = 0; i < label_size; i++) {
        labels[i] = 0;  // set all labels to 0
    }
}
