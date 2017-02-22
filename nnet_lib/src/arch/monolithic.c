#include "nnet_fwd.h"
#include "core/activation_functions.h"
#include "core/convolution.h"
#include "core/matrix_multiply.h"
#include "core/pooling.h"
#include "core/zeropad.h"
#include "utility/utility.h"
#include "arch/common.h"
#include "arch/interface.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

#if ARCHITECTURE == MONOLITHIC

// This is an architecture that runs an entire neural network in a single
// block, where nnet_fwd is the top level function. nnet_fwd is thus
// responsible for ensuring that all input and weights data is available when
// each layer needs them.

result_buf inner_product_layer(float* activations,
                               float* weights,
                               layer_t* layers,
                               int lnum,
                               float* result) {
    MATRIX_MULTIPLY_WITH_BIAS(activations, weights, NUM_TEST_CASES,
                              layers[lnum].input_rows, layers[lnum].input_cols,
                              result);
    return result;
}

result_buf convolution_layer(float* input,
                             float* kernels,
                             layer_t* layers,
                             int lnum,
                             float* result) {
    layer_t curr_layer = layers[lnum];
    if (curr_layer.c_padding > 0) {
        convolution2d_zeropad(input, kernels, curr_layer, result);
        return input;
    }
    convolution2d_no_padding(input, kernels, curr_layer, result);
    return result;
}

result_buf max_pooling_layer(float* input,
                             layer_t* layers,
                             int lnum,
                             float* result) {
    layer_t curr_layer = layers[lnum];
    max_pooling(input, result, curr_layer);
    return result;
}

// Runs the forward pass of a neural network.
//
// This version loads weights on a per layer basis, and activations are
// ping-ponged between two buffers, input and result.
void nnet_fwd(float* input,
              float* weights,
              layer_t* layers,
              int num_layers,
              float* result,
              float* sigmoid_table) {

    int l;
    layer_t curr_layer;

    // Alternate between reading from/writing to input and result so we can
    // avoid copying matrices. The initial input is obviously in "input",
    // so that's where we start.
    result_buf result_loc = input;
    bool do_activation_func = true;

    if (PRINT_DATA_AND_WEIGHTS) {
        print_data_and_weights(input, weights, layers[0]);
    }

    // FORMAT HERE IS H TIMES W, NOT W TIMES H!!!!!
    // SO EACH DATA POINT IS A ***ROW****

    l = 0;
    dmaLoad(input, 0, 0, NUM_TEST_CASES * INPUT_DIM * sizeof(float));

    //******************//
    //   PRIMARY LOOP   //
    //******************//

nnet_fwd_outer:
    for (l = 0; l < num_layers; l++) {
        curr_layer = layers[l];
        // Don't run the activation function on the last layer.
        do_activation_func = (l != num_layers - 1);

        grab_matrix_dma(weights, l, layers);

        if (result_loc == result) {
            result_loc = run_layer(result, weights, layers, l, input,
                                   sigmoid_table, do_activation_func);
        } else {
            result_loc = run_layer(input, weights, layers, l, result,
                                   sigmoid_table, do_activation_func);
        }
    }

    layers[num_layers - 1].result_in_temp = result_loc == result;

    if (result_loc == result)
        dmaStore(result, 0, 0, NUM_TEST_CASES * NUM_CLASSES * sizeof(float));
    else
        dmaStore(input, 0, 0, NUM_TEST_CASES * NUM_CLASSES * sizeof(float));
    dmaStore(layers, 0, 0, num_layers*sizeof(layer_t));
}

#endif
