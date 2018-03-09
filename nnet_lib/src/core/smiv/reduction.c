#include <assert.h>
#include <stdio.h>

#include "core/ref/activation_functions.h"
#include "core/smiv/activation_functions_simd.h"
#include "utility/utility.h"
#include "nnet_fwd.h"

#include "impls.h"

void reduction_smiv_fxp(float* a, layer_t curr_layer, float* result) {
    int row, col, chan, c;

    const int result_height = curr_layer.outputs.rows;
    const int result_width = curr_layer.outputs.cols;
    const int result_pad = curr_layer.outputs.align_pad;
    const int padded_width = result_width + result_pad;

    const int k_height =  curr_layer.inputs.height;
    const bool run_activation = curr_layer.activation != NO_ACTIVATION;

#ifdef TRACE_MODE
    assert(padded_width % VECTOR_SIZE == 0 &&
           "Padded width must be multiple of VECTOR_SIZE!");
#endif

    ARRAY_3D(float, _a, a, result_height, padded_width);
    ARRAY_2D(float, _result, result, padded_width);

    reduction_row:
    for (row = 0; row < result_height; row++) {
        PRINT_MSG_V("Reduction of row %d\n", row);
        reduction_col:
        for (col = 0; col < padded_width; col += VECTOR_SIZE) {
            PRINT_MSG_V("Col %d\n  ", col);
            float partial_sums[VECTOR_SIZE] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            reduction_chan:
            for (chan = 0; chan < k_height; chan++) {
                reduction_core:
                for (c = 0; c < VECTOR_SIZE; c++) {
                    partial_sums[c] += _a[chan][row][col + c];
                    PRINT_MSG_V("%9.5f\t", _a[chan][row][col + c]);
                }
                PRINT_MSG_V("\n  ");
            }

            PRINT_MSG_V("---------------\n  ");
            PRINT_DEBUG_V(&partial_sums[0], 1, VECTOR_SIZE, VECTOR_SIZE);

            if (run_activation) {
                activation_fun(&partial_sums[0],
                               1,
                               VECTOR_SIZE,
                               result_pad,
                               curr_layer.activation);
            }

            reduction_commit:
            for (c = 0; c < VECTOR_SIZE; c++) {
                _result[row][col + c] = partial_sums[c];
            }
        }
    }
}

void reduction_smiv_vec_fxp(float* a, layer_t curr_layer, float* result) {
    int row, col, chan;

    const int result_height = curr_layer.outputs.rows;
    const int result_width = curr_layer.outputs.cols;
    const int result_pad = curr_layer.outputs.align_pad;
    const int padded_width = result_width + result_pad;
    const int vec_padded_width = padded_width / VECTOR_SIZE;

    const int k_height =  curr_layer.inputs.height;
    const bool run_activation = curr_layer.activation != NO_ACTIVATION;

#ifdef TRACE_MODE
    assert(padded_width % VECTOR_SIZE == 0 &&
           "Padded width must be multiple of VECTOR_SIZE!");
#endif

    VEC_ARRAY_3D(v8fp_t, _a, a, result_height, padded_width);
    VEC_ARRAY_2D(v8fp_t, _result, result, padded_width);

    reduction_row:
    for (row = 0; row < result_height; row++) {
        reduction_col:
        for (col = 0; col < vec_padded_width; col++) {
            v8fp_t partial_sums;
            partial_sums = (v8fp_t){ 0, 0, 0, 0, 0, 0, 0, 0 };
            reduction_chan:
            for (chan = 0; chan < k_height; chan++) {
                partial_sums += _a[chan][row][col];
            }

            if (run_activation) {
                partial_sums = activation_fun_simd_fxp(
                        partial_sums, curr_layer.activation);
            }

            _result[row][col] = partial_sums;
        }
    }
}
