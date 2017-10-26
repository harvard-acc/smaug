#include <assert.h>

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

#if ARCHITECTURE == COMPOSABLE

unsigned kConvolutionHw = 0x0001;
unsigned kPoolingHw = 0x0002;
unsigned kActivationFuncHw = 0x0003;
unsigned kInnerProductHw = 0x0004;

// This is an architecture that divides each layer type into a separate
// hardware block. This is represented by ensuring that each layer is
// responsible for loading its own input activations and weights. For clarity,
// all functions to be turned into hardware are suffixed with _hw.

void inner_product_layer_hw(float* activations,
                            float* weights,
                            layer_t* layers,
                            int lnum,
                            float* result) {
    grab_weights_dma(weights, lnum, layers);
    grab_input_activations_dma(activations, lnum, layers);
    MATRIX_MULTIPLY_WITH_BIAS(
            activations, weights, NUM_TEST_CASES, layers[lnum].weights.rows,
            layers[lnum].weights.cols + layers[lnum].weights.align_pad,
            result);
    store_output_activations_dma(result, lnum, layers);
}

result_buf inner_product_layer(float* activations,
                               float* weights,
                               layer_t* layers,
                               int lnum,
                               float* result) {
    MAP_ARRAY(kInnerProductHw, activations, INPUT_BYTES(layers, lnum));
    MAP_ARRAY(kInnerProductHw, weights, OUTPUT_BYTES(layers, lnum));
    MAP_ARRAY(kInnerProductHw, result, WEIGHT_BYTES(layers, lnum));
    INVOKE_KERNEL(kInnerProductHw, inner_product_layer_hw, activations, weights,
                  layers, lnum, result);
    return result;
}

void convolution_layer_hw(float* activations,
                          float* weights,
                          layer_t* layers,
                          int lnum,
                          float* result) {
    layer_t curr_layer = layers[lnum];
    grab_weights_dma(weights, lnum, layers);
    grab_input_activations_dma(activations, lnum, layers);
    convolution2d_no_padding(activations, weights, curr_layer, result);
    store_output_activations_dma(result, lnum, layers);
}

result_buf convolution_layer(float* activations,
                             float* weights,
                             layer_t* layers,
                             int lnum,
                             float* result) {
    MAP_ARRAY(kConvolutionHw, activations, INPUT_BYTES(layers, lnum));
    MAP_ARRAY(kConvolutionHw,  weights, OUTPUT_BYTES(layers, lnum));
    MAP_ARRAY(kConvolutionHw,  result, WEIGHT_BYTES(layers, lnum));

    layer_t curr_layer = layers[lnum];
    if (curr_layer.c_padding > 0) {
        // TODO: Replace this with a memcpy implementation.
        copy_zeropad(activations, layers, lnum, result);
        INVOKE_KERNEL(kConvolutionHw, convolution_layer_hw, result, weights,
                      layers, lnum, activations);

        return activations;
    }
    INVOKE_KERNEL(kConvolutionHw, convolution_layer_hw, activations, weights, layers,
                  lnum, result);
    return result;
}

void max_pooling_layer_hw(float* activations,
                          float* result,
                          layer_t* layers,
                          int lnum) {
    layer_t curr_layer = layers[lnum];
    grab_input_activations_dma(activations, lnum, layers);
    max_pooling(activations, result, curr_layer);
    store_output_activations_dma(result, lnum, layers);
}

result_buf pooling_layer(float* activations,
                         layer_t* layers,
                         int lnum,
                         float* result) {
    layer_t curr_layer = layers[lnum];
    MAP_ARRAY(kPoolingHw, activations, INPUT_BYTES(layers, lnum));
    MAP_ARRAY(kPoolingHw, result, OUTPUT_BYTES(layers, lnum));
    if (curr_layer.pool == MAX) {
        INVOKE_KERNEL(kPoolingHw, max_pooling_layer_hw, activations, result,
                      layers, lnum);
    } else {
        assert(false && "Unsupported pooling layer type!");
    }
    return result;
}

void activation_hw(float* activations,
                   layer_t* layers,
                   int lnum,
                   float* sigmoid_table) {
    layer_t curr_layer = layers[lnum];
    int size = grab_output_activations_dma(activations, lnum, layers);
    activation_fun(activations, size, curr_layer.activation, sigmoid_table);
    store_output_activations_dma(activations, lnum, layers);
}

result_buf activation_sublayer(float* activations,
                               layer_t* layers,
                               int lnum,
                               float* sigmoid_table) {
    MAP_ARRAY(kActivationFuncHw, activations, OUTPUT_BYTES(layers, lnum));
    INVOKE_KERNEL(kActivationFuncHw, activation_hw, activations, layers, lnum,
                  sigmoid_table);
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

// Runs the forward pass of a neural network.
//
// This version loads weights on a per layer basis, and activations are
// ping-ponged between two buffers, activations and result.
void nnet_fwd(farray_t activations,
              farray_t weights,
              farray_t result,
              network_t network,
              float* sigmoid_table) {

    int l;
    layer_t curr_layer;

    // Alternate between reading from/writing to activations and result so we
    // can avoid copying matrices. The initial activations is obviously in
    // "activations", so that's where we start.
    result_buf result_loc = activations.d;

    if (PRINT_DATA_AND_WEIGHTS) {
        print_data_and_weights(activations.d, weights.d, network.layers[0]);
    }

    // FORMAT HERE IS H TIMES W, NOT W TIMES H!!!!!
    // SO EACH DATA POINT IS A ***ROW****

    l = 1;

    //******************//
    //   PRIMARY LOOP   //
    //******************//

nnet_fwd_outer:
    for (l = 1; l < network.depth; l++) {
        curr_layer = network.layers[l];

        if (result_loc == result.d) {
            result_loc = run_layer(result.d, weights.d, network.layers, l,
                                   activations.d, sigmoid_table);
        } else {
            result_loc = run_layer(activations.d, weights.d, network.layers, l,
                                   result.d, sigmoid_table);
        }
    }

    network.layers[network.depth - 1].result_in_temp = (result_loc == result.d);

    if (result_loc == result.d)
        dmaStore(result.d, 0, 0, NUM_TEST_CASES * NUM_CLASSES * sizeof(float));
    else
        dmaStore(activations.d, 0, 0,
                 NUM_TEST_CASES * NUM_CLASSES * sizeof(float));
    dmaStore(network.layers, 0, 0, network.depth * sizeof(layer_t));
}

#endif
