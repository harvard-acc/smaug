#include "nnet_fwd.h"
#include "core/activation_functions.h"
#include "core/convolution.h"
#include "core/matrix_multiply.h"
#include "core/pooling.h"
#include "core/zeropad.h"
#include "utility/utility.h"
#include "monolithic.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

bool run_layer_m(float* activations,
                 float* weights,
                 layer_t curr_layer,
                 float* result_temp,
                 float* sigmoid_table,
                 bool do_activation_func) {
    bool result_in_input = false;
    layer_type l_type = curr_layer.type;
    if (l_type == FC) {
        PRINT_MSG("\nmatrix multiply with bias\n");
        MATRIX_MULTIPLY_WITH_BIAS(activations, weights, NUM_TEST_CASES,
                                  curr_layer.input_rows, curr_layer.input_cols,
                                  result_temp);
        PRINT_DEBUG4D(result_temp, curr_layer.output_rows,
                      curr_layer.output_cols, curr_layer.output_height);
    } else if (l_type == CONV) {
        PRINT_MSG("\nconvolution2d\n");
        if (curr_layer.c_padding > 0) {
            convolution2d_zeropad(
                    activations, weights, curr_layer, result_temp);
            PRINT_DEBUG4D(activations, curr_layer.output_rows,
                          curr_layer.output_cols, curr_layer.output_height);
            result_in_input = true;
        } else {
            convolution2d_no_padding(
                    activations, weights, curr_layer, result_temp);
            PRINT_DEBUG4D(result_temp, curr_layer.output_rows,
                          curr_layer.output_cols, curr_layer.output_height);
        }
    } else if (l_type == POOL_MAX) {
        PRINT_MSG("\nmax pooling\n");
        max_pooling(activations, result_temp, curr_layer);
        PRINT_DEBUG4D(result_temp, curr_layer.output_rows, curr_layer.output_cols,
                      curr_layer.output_height);
    }

    if (do_activation_func) {
        PRINT_MSG("\nactivation function\n");
        // Pass through activation function
        if (result_in_input) {
            activation_fun(activations,
                           curr_layer.output_rows * curr_layer.output_cols *
                                   curr_layer.output_height,
                           sigmoid_table);

            PRINT_DEBUG4D(activations, curr_layer.output_rows,
                          curr_layer.output_cols, curr_layer.output_height);
        } else {
            activation_fun(result_temp,
                           curr_layer.output_rows * curr_layer.output_cols *
                                   curr_layer.output_height,
                           sigmoid_table);

            PRINT_DEBUG4D(result_temp, curr_layer.output_rows,
                          curr_layer.output_cols, curr_layer.output_height);
        }

    }
    return result_in_input;
}

// Runs the forward pass of a neural network.
//
// This version loads weights on a per layer basis, and activations are
// ping-ponged between two buffers, hid and hid_temp.
void nnet_fwd_monolithic(float* hid,
                         float* weights,
                         layer_t* layers,
                         int num_layers,
                         float* hid_temp,
                         float* sigmoid_table) {

    int l;
    layer_t curr_layer;

    // Alternate between reading from/writing to hid and hid_temp so we can
    // avoid copying matrices.
    bool result_in_temp = false;
    bool result_in_input = false;
    bool do_activation_func = true;

    if (PRINT_DATA_AND_WEIGHTS) {
        print_data_and_weights(hid, weights, layers[0]);
    }

    // FORMAT HERE IS H TIMES W, NOT W TIMES H!!!!!
    // SO EACH DATA POINT IS A ***ROW****

    l = 0;
    dmaLoad(hid, 0, 0, NUM_TEST_CASES * INPUT_DIM * sizeof(float));

    //******************//
    //   PRIMARY LOOP   //
    //******************//

nnet_fwd_outer:
    for (l = 0; l < num_layers; l++) {
        curr_layer = layers[l];
        // Don't run the activation function on the last layer.
        do_activation_func = (l != num_layers - 1);

        grab_matrix_dma(weights, l, layers);

        if (result_in_temp) {
            result_in_input = run_layer_m(hid_temp, weights, curr_layer, hid,
                                          sigmoid_table, do_activation_func);
        } else {
            result_in_input = run_layer_m(hid, weights, curr_layer, hid_temp,
                                          sigmoid_table, do_activation_func);
        }

        if (!result_in_input)
           result_in_temp = !result_in_temp;
    }

    layers[num_layers - 1].result_in_temp = (int)result_in_temp;

    if (result_in_temp)
        dmaStore(hid_temp, 0, 0, NUM_TEST_CASES * NUM_CLASSES * sizeof(float));
    else
        dmaStore(hid, 0, 0, NUM_TEST_CASES * NUM_CLASSES * sizeof(float));
    dmaStore(layers, 0, 0, num_layers*sizeof(layer_t));
}
