#include <assert.h>
#include <stdio.h>

#include "core/activation_functions.h"
#include "utility/utility.h"
#include "nnet_fwd.h"

#include "impls.h"

void reduction_smiv_fxp(
        float* a, layer_t curr_layer, int img, int kern, float* result) {
    unsigned row, col, chan, c;

    const int result_height = curr_layer.outputs.rows;
    const int result_width = curr_layer.outputs.cols;
    const int result_pad = curr_layer.outputs.align_pad;
    const int padded_width = result_width + result_pad;

    const int k_height =  curr_layer.inputs.height;
    const int num_kerns = curr_layer.outputs.height;
    const bool run_activation = curr_layer.activation != NONE;

#ifdef TRACE_MODE
    assert(padded_width % VECTOR_SIZE == 0 &&
           "Padded width must be multiple of VECTOR_SIZE!");
#endif

    ARRAY_3D(float, _a, a, result_height, padded_width);
    ARRAY_4D(float, _result, result, num_kerns, result_height, padded_width);

    reduction_row:
    for (row = 0; row < result_height; row++) {
        reduction_col:
        for (col = 0; col < padded_width; col += VECTOR_SIZE) {
            float partial_sums[VECTOR_SIZE] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            reduction_chan:
            for (chan = 0; chan < k_height; chan++) {
                reduction_core:
                for (c = 0; c < VECTOR_SIZE; c++) {
                    partial_sums[c] += _a[chan][row][col + c];
                }
            }

            if (run_activation) {
                activation_fun(&partial_sums[0], VECTOR_SIZE, RELU, NULL);
            }

            reduction_commit:
            for (c = 0; c < VECTOR_SIZE; c++) {
                _result[img][kern][row][col + c] = partial_sums[c];
            }
        }
    }
}

void reduction_smiv_vec_fxp(
        float* a, layer_t curr_layer, int img, int kern, float* result) {
    unsigned row, col, chan;

    const int result_height = curr_layer.outputs.rows;
    const int result_width = curr_layer.outputs.cols;
    const int result_pad = curr_layer.outputs.align_pad;
    const int padded_width = result_width + result_pad;
    const int vec_padded_width = padded_width / VECTOR_SIZE;

    const int k_height =  curr_layer.inputs.height;
    const int num_kerns = curr_layer.outputs.height;
    const bool run_activation = curr_layer.activation != NONE;

#ifdef TRACE_MODE
    assert(padded_width % VECTOR_SIZE == 0 &&
           "Padded width must be multiple of VECTOR_SIZE!");
#endif

    VEC_ARRAY_3D(v8fp_t, _a, a, result_height, padded_width);
    VEC_ARRAY_4D(v8fp_t, _result, result, num_kerns, result_height, padded_width);

    reduction_row:
    for (row = 0; row < result_height; row++) {
        reduction_col:
        for (col = 0; col < vec_padded_width; col++) {
            v8fp_t partial_sums = (v8fp_t){ 0, 0, 0, 0, 0, 0, 0, 0 };
            reduction_chan:
            for (chan = 0; chan < k_height; chan++) {
                partial_sums += _a[chan][row][col];
            }

            if (run_activation) {
                RELU_VEC_SMIV(partial_sums);
            }

            _result[img][kern][row][col] = partial_sums;
        }
    }
}
