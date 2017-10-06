#include <assert.h>

#include "core/activation_functions.h"
#include "utility/utility.h"
#include "nnet_fwd.h"

#include "impls.h"

// A = activations
// B = weights
// B must NOT be transposed!
//
// This works on a batch of inputs, reusing the model weights for each input activation.
void matrix_multiply_with_bias_smiv_batch_fxp(float* a,
                                              float* b,
                                              int a_height,
                                              int b_height,
                                              int b_width,
                                              bool run_activation,
                                              float* result) {
    int wgt_row, wgt_col, wgt_b;
    int act_batch;
    float partial_sums[MAX_BATCH][VECTOR_SIZE];
    float input, weight, product, bias;

    int a_width = b_height - 1;

    ARRAY_2D(float, _a, a, a_width);
    ARRAY_2D(float, _b, b, b_width);
    ARRAY_2D(float, _result, result, b_width);

#ifndef TRACE_MODE
    assert(b_width % VECTOR_SIZE == 0 &&
           "Width of weights must be a multiple of VECTOR_SIZE!");
#endif

    wgt_col:
    for (wgt_col = 0; wgt_col < b_width; wgt_col+=VECTOR_SIZE) {
        // Load in the bias.
        load_bias_batch:
        for (act_batch = 0; act_batch < a_height; act_batch++) {
            load_bias:
            for (wgt_b = 0; wgt_b < VECTOR_SIZE; wgt_b++) {
                bias = conv_float2fixed(_b[a_width][wgt_col + wgt_b]);
                partial_sums[act_batch][wgt_b] = bias;
            }
        }

        wgt_row:
        for (wgt_row = 0; wgt_row < a_width; wgt_row++) {
            act_batch_macc:
            for (act_batch = 0; act_batch < a_height; act_batch++) {
                // MACC datapath.
                // Flatten this inner loop.
                wgt_b_macc:
                for (wgt_b = 0; wgt_b < VECTOR_SIZE; wgt_b++) {
                    input = conv_float2fixed(_a[act_batch][wgt_row]);
                    weight = conv_float2fixed(_b[wgt_row][wgt_col + wgt_b]);

                    product = input * weight;
                    partial_sums[act_batch][wgt_b] += product;
                }
            }
        }

        // Run through activation function.
        if (run_activation) {
            run_activation_func:
            for (act_batch = 0; act_batch < a_height; act_batch++)
                activation_fun(&partial_sums[act_batch][0], VECTOR_SIZE, RELU, NULL);
        }

        // Store to scratchpad.
        act_batch_store:
        for (act_batch = 0; act_batch < a_height; act_batch++) {
            wgt_b_store:
            for (wgt_b = 0; wgt_b < VECTOR_SIZE; wgt_b++) {
                _result[act_batch][wgt_col + wgt_b] = partial_sums[act_batch][wgt_b];
            }
        }
    }
}

// A = activations
// B = weights
// B must NOT be transposed!
//
// This works on a single input at a time, so it does not store partial sums
// for more than one input vector and does not exploit weight reuse.
//
// This implementation lets Aladdin do a better job of imitating performance
// when working on one input (to avoid a useless layer of inner looping that
// introduces additional overheads).
void matrix_multiply_with_bias_smiv_nobatch_fxp(float* a,
                                                float* b,
                                                int a_height,
                                                int b_height,
                                                int b_width,
                                                bool run_activation,
                                                float* result) {
    int wgt_row, wgt_col, wgt_b, input_act;
    float partial_sums[VECTOR_SIZE];
    float input, weight, product, bias;

    int a_width = b_height - 1;

    ARRAY_2D(float, _a, a, a_width);
    ARRAY_2D(float, _b, b, b_width);
    ARRAY_2D(float, _result, result, b_width);

#ifndef TRACE_MODE
    assert(b_width % VECTOR_SIZE == 0 &&
           "Width of weights must be a multiple of VECTOR_SIZE!");
#endif

    input_act:
    for (input_act = 0; input_act < a_height; input_act++) {
        wgt_col:
        for (wgt_col = 0; wgt_col < b_width; wgt_col+=VECTOR_SIZE) {
            // Load in the bias.
            load_bias:
            for (wgt_b = 0; wgt_b < VECTOR_SIZE; wgt_b++) {
                bias = conv_float2fixed(_b[a_width][wgt_col + wgt_b]);
                partial_sums[wgt_b] = bias;
            }

            wgt_row:
            for (wgt_row = 0; wgt_row < a_width; wgt_row++) {
                // MACC datapath.
                // Flatten this inner loop.
                wgt_b_macc:
                for (wgt_b = 0; wgt_b < VECTOR_SIZE; wgt_b++) {
                    input = conv_float2fixed(_a[input_act][wgt_row]);
                    weight = conv_float2fixed(_b[wgt_row][wgt_col + wgt_b]);

                    product = input * weight;
                    partial_sums[wgt_b] += product;
                }
            }

            // Run through activation function.
            if (run_activation) {
                activation_fun(&partial_sums[0], VECTOR_SIZE, RELU, NULL);
            }

            // Store to scratchpad.
            wgt_b_store:
            for (wgt_b = 0; wgt_b < VECTOR_SIZE; wgt_b++) {
                _result[input_act][wgt_col + wgt_b] = partial_sums[wgt_b];
            }
        }
    }
}

