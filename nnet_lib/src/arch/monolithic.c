#include <assert.h>

#include "arch/common.h"
#include "arch/interface.h"
#include "core/activation_functions.h"
#include "core/convolution.h"
#include "core/matrix_multiply.h"
#include "core/pooling.h"
#include "core/zeropad.h"
#include "nnet_fwd.h"
#include "utility/utility.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

#if ARCHITECTURE == MONOLITHIC

unsigned kNnetFwdHw = 0x0001;

// This is an architecture that runs an entire neural network in a single
// block, where nnet_fwd is the top level function. nnet_fwd is thus
// responsible for ensuring that all activations and weights data is available
// when
// each layer needs them.

result_buf inner_product_layer(float* activations,
                               float* weights,
                               layer_t* layers,
                               int lnum,
                               float* result) {
    PRINT_MSG("Weights:\n");
    PRINT_DEBUG(weights, layers[lnum].weights.rows, layers[lnum].weights.cols,
                layers[lnum].weights.cols + layers[lnum].weights.align_pad);
    MATRIX_MULTIPLY_WITH_BIAS(
            activations, weights, NUM_TEST_CASES, layers[lnum].weights.rows,
            layers[lnum].weights.cols + layers[lnum].weights.align_pad,
            result);
    return result;
}

result_buf convolution_layer(float* activations,
                             float* kernels,
                             layer_t* layers,
                             int lnum,
                             float* result) {
    layer_t curr_layer = layers[lnum];
    if (curr_layer.c_padding > 0) {
        convolution2d_zeropad(activations, kernels, layers, lnum, result);
        return activations;
    }
    convolution2d_no_padding(activations, kernels, curr_layer, result);
    return result;
}

result_buf pooling_layer(float* activations,
                         layer_t* layers,
                         int lnum,
                         float* result) {
    layer_t curr_layer = layers[lnum];
    if (curr_layer.pool == MAX)
        max_pooling(activations, result, curr_layer);
    else
        assert(false && "Unsupported pooling layer type!");
    return result;
}

result_buf activation_sublayer(float* activations,
                               layer_t* layers,
                               int lnum,
                               float* sigmoid_table) {
    int size = get_output_activations_size(&layers[lnum]);
    activation_fun(activations, size, layers[lnum].activation, sigmoid_table);
    return activations;
}

result_buf run_layer(float* activations,
                     float* weights,
                     layer_t* layers,
                     int layer_num,
                     float* result,
                     float* sigmoid_table) {
    layer_t curr_layer = layers[layer_num];
    result_buf result_loc = run_layer_skip_activation_func(
            activations, weights, layers, layer_num, result, sigmoid_table);

    if (curr_layer.activation != NONE) {
        PRINT_MSG("\nactivation function\n");
        // Pass through activation function
        if (result_loc == activations) {
            activation_sublayer(activations, layers, layer_num, sigmoid_table);
        } else {
            activation_sublayer(result, layers, layer_num, sigmoid_table);
        }

        PRINT_DEBUG4D(result_loc, curr_layer.outputs.rows,
                      curr_layer.outputs.cols + curr_layer.outputs.align_pad,
                      curr_layer.outputs.height);
    }
    return result_loc;
}

void nnet_fwd_hw(float* activations,
                 float* weights,
                 layer_t* layers,
                 int num_layers,
                 float* result,
                 float* sigmoid_table) {
    int l;
    layer_t curr_layer;

    // Alternate between reading from/writing to activations and result so we
    // can
    // avoid copying matrices. The initial activations is obviously in
    // "activations",
    // so that's where we start.
    result_buf result_loc = activations;

    // FORMAT HERE IS H TIMES W, NOT W TIMES H!!!!!
    // SO EACH DATA POINT IS A ***ROW****

    l = 0;
    dmaLoad(activations, activations, NUM_TEST_CASES * INPUT_DIM * sizeof(float));

    //******************//
    //   PRIMARY LOOP   //
    //******************//

nnet_fwd_outer:
    for (l = 1; l < num_layers; l++) {
        curr_layer = layers[l];

        grab_weights_dma(weights, weights, l, layers);

        if (result_loc == result) {
            result_loc = run_layer(
                    result, weights, layers, l, activations, sigmoid_table);
        } else {
            result_loc = run_layer(
                    activations, weights, layers, l, result, sigmoid_table);
        }
    }

    layers[num_layers - 1].result_in_temp = result_loc == result;

    if (result_loc == result)
        dmaStore(result, result, NUM_TEST_CASES * NUM_CLASSES * sizeof(float));
    else
        dmaStore(activations, activations,
                 NUM_TEST_CASES * NUM_CLASSES * sizeof(float));
    dmaStore(layers, layers, num_layers * sizeof(layer_t));
}


// Runs the forward pass of a neural network.
//
// This version loads weights on a per layer basis, and activations are
// ping-ponged between two buffers, activations and result.
void nnet_fwd(farray_t activations,
              farray_t weights,
              farray_t result,
              network_t network,
              float* sigmoid_table) {
    if (PRINT_DATA_AND_WEIGHTS) {
        print_data_and_weights(activations.d, weights.d, network.layers[0]);
    }

    MAP_ARRAY_TO_ACCEL(kNnetFwdHw, "activations", activations.d,
                       activations.size * sizeof(float));
    MAP_ARRAY_TO_ACCEL(
            kNnetFwdHw, "weights", weights.d, weights.size * sizeof(float));
    MAP_ARRAY_TO_ACCEL(
            kNnetFwdHw, "result", result.d, result.size * sizeof(float));
    MAP_ARRAY_TO_ACCEL(kNnetFwdHw, "layers", network.layers,
                       network.depth * sizeof(layer_t));

    INVOKE_KERNEL(kNnetFwdHw, nnet_fwd_hw, activations.d, weights.d, network.layers,
                  network.depth, result.d, sigmoid_table);
}

#endif
