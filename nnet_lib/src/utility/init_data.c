#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

#include "utility/utility.h"
#include "nnet_fwd.h"

#include "init_data.h"

void init_weights(float* weights,
                  layer_t* layers,
                  int num_layers,
                  data_init_mode mode,
                  bool transpose) {
    int d, h, i, j, l;
    int w_rows, w_cols, w_height, w_depth, w_offset, w_pad;
    float val;

    assert(mode == RANDOM || mode == FIXED);
    w_offset = 0;
    printf("Initializing weights randomly\n");

    for (l = 0; l < num_layers; l++) {
        get_weights_dims_layer(
                layers, l, &w_rows, &w_cols, &w_height, &w_depth, &w_pad);
        int w_tot_cols = w_cols + w_pad;
        for (d = 0; d < w_depth; d++) {
            for (h = 0; h < w_height; h++) {
                for (i = 0; i < w_rows; i++) {
                    for (j = 0; j < w_tot_cols; j++) {
                        if (j < w_cols) {
                            if (mode == RANDOM) {
                                val = conv_float2fixed(
                                        (randfloat() - 0.5) *
                                        10);  // Question: does nan output
                                              // take longer in simulation?
                            } else {
                                // Give each depth slice a different weight
                                // so
                                // we don't get all the same value in the
                                // output.
                                val = (d + 1) * 0.1;
                            }
                        } else {  // extra zero padding.
                            val = 0;
                        }
                        // Use the subxind macros here instead of
                        // multidimensional array indexing because the
                        // dimensionality of the weights varies within this
                        // single function.
                        if (layers[l].type == FC) {
                            if (transpose)
                                weights[sub3ind(h, j, i, w_tot_cols, w_rows) +
                                        w_offset] = val;
                            else
                                weights[sub3ind(h, i, j, w_rows, w_tot_cols) +
                                        w_offset] = val;
                        } else if (layers[l].type == BATCH_NORM &&
                                   (i / (w_rows / 4) == 1) && val < 0) {
                            // For batch norm layer, the weights' rows are
                            // organized as {mean, var, gamma, beta}. var
                            // should not have negative values.
                            weights[sub4ind(d, h, i, j, w_height, w_rows,
                                            w_tot_cols) +
                                    w_offset] = -val;
                        } else {
                            weights[sub4ind(d, h, i, j, w_height, w_rows,
                                            w_tot_cols) +
                                    w_offset] = val;
                        }
                    }
                }
            }
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
        for (j = 0; j < input_height; j++) {
            for (k = 0; k < input_rows; k++) {
                for (l = 0; l < input_cols; l++) {
                    if (mode == RANDOM) {
                        _data[i][j][k][l] = conv_float2fixed(randfloat() - 0.5);
                    } else {
                        // Make each input image distinguishable.
                        unsigned long addr =
                                (unsigned long)(&_data[i][j][k][l]);
                        addr -= (unsigned long)data;
                        unsigned offset = addr / sizeof(float);
                        _data[i][j][k][l] = 1.0 * i + (float)offset / input_dim;
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
