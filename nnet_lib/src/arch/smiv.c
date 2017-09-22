#include <assert.h>

#include "nnet_fwd.h"
#include "core/activation_functions.h"
#include "core/convolution.h"
#include "core/matrix_multiply.h"
#include "core/pooling.h"
#include "core/smiv_core.h"
#include "core/zeropad.h"
#include "utility/utility.h"
#include "arch/common.h"
#include "arch/interface.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

#if ARCHITECTURE == SMIV

// Use the same accelerator id for both the convolutional and FC blocks. This
// means we will simulate only ONE datapath instead of two, which means that
// the two blocks can share the scratchpads (without any more infrastructure
// changes). The key is that we still trace the functions at the _hw level, so
// that Aladdin will exit after simulating each block, and we can return
// control to the CPU at the right places.  In contrast, if we used two
// different ids, we would have two different datapaths that could not share
// data directly.
unsigned kConvolutionHw = 0x0003;
unsigned kInnerProductHw = 0x0003;

void inner_product_layer_hw(float* activations,
                            float* weights,
                            layer_t* layers,
                            int lnum,
                            float* result) {
    bool run_activation = layers[lnum].activation != NONE;
    grab_matrix_dma(weights, lnum, layers);
    if (layers[lnum].needs_input_dma_load)
        grab_input_activations_dma(activations, lnum, layers);
    matrix_multiply_with_bias_smiv(
            activations, weights, NUM_TEST_CASES, layers[lnum].input_rows,
            layers[lnum].input_cols + layers[lnum].input_data_align_pad,
            run_activation, result);
    if (layers[lnum].needs_output_dma_store)
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
    grab_matrix_dma(weights, lnum, layers);
    if (layers[lnum].needs_input_dma_load)
        grab_input_activations_dma(activations, lnum, layers);
    convolution2d_smiv(activations, weights, curr_layer, result);
    if (layers[lnum].needs_output_dma_store)
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
        int padding = (curr_layer.field_size - 1) / 2;
        // TODO: Replace this with a memcpy implementation.
        copy_zeropad(activations, curr_layer, padding, result);
        PRINT_MSG("After zeropadding:\n");
        PRINT_DEBUG4D(result,
                      curr_layer.input_rows,
                      curr_layer.input_cols + curr_layer.input_data_align_pad,
                      curr_layer.input_height);
        INVOKE_KERNEL(kConvolutionHw, convolution_layer_hw, result, weights,
                      layers, lnum, activations);

        return activations;
    }
    INVOKE_KERNEL(kConvolutionHw, convolution_layer_hw, activations, weights, layers,
                  lnum, result);
    return result;
}

// Software implementation. SMIV doesn't accelerate pooling.
result_buf pooling_layer(float* activations,
                         layer_t* layers,
                         int lnum,
                         float* result) {
    layer_t curr_layer = layers[lnum];
    if (curr_layer.pool == MAX) {
        max_pooling(activations, result, layers[lnum]);
    } else {
        assert(false && "Unsupported pooling layer type!");
    }
    return result;
}

result_buf run_layer(float* activations,
                     float* weights,
                     layer_t* layers,
                     int layer_num,
                     float* result,
                     float* sigmoid_table) {
    result_buf result_loc = run_layer_skip_activation_func(
            activations, weights, layers, layer_num, result, sigmoid_table);

    // Activation functions are handled as part of the matrix multiply /
    // convolution, rather than being treated as a separate block.
    return result_loc;
}

// Set the dmaLoad/dmaStore required flags for each layer.
//
// Since SMIV can share scratchpads between the conv/fc blocks, we only need
// DMA if we need to send data back to the CPU.
void set_dma_requirements(network_t* network) {
    for (int layer_num = 0; layer_num < network->depth; layer_num++) {
        // The input layer is easy.
        if (layer_num == 0) {
            network->layers[layer_num].needs_input_dma_load = false;
            network->layers[layer_num].needs_output_dma_store = true;
            continue;
        }
        // First, determine if we need to dma store the output.
        if (layer_num == network->depth - 1 ||
            network->layers[layer_num].activation == SIGMOID ||
            network->layers[layer_num].type == POOLING ||
            network->layers[layer_num].flatten_input ||
            network->layers[layer_num + 1].type == POOLING ||
            network->layers[layer_num + 1].type == SOFTMAX) {
            network->layers[layer_num].needs_output_dma_store = true;
        } else {
            network->layers[layer_num].needs_output_dma_store = false;
        }

        // Whether we need to load the input on this layer is just whether we
        // had to store the outputs in the previous layer.
        network->layers[layer_num].needs_input_dma_load =
                network->layers[layer_num - 1].needs_output_dma_store;
    }

    for (int layer_num = 0; layer_num < network->depth; layer_num++) {
        printf("Layer %d: dmaLoad = %d, dmaStore = %d\n", layer_num,
               network->layers[layer_num].needs_input_dma_load,
               network->layers[layer_num].needs_output_dma_store);
    }

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

    l = 0;

    set_dma_requirements(&network);

    //******************//
    //   PRIMARY LOOP   //
    //******************//

nnet_fwd_outer:
    for (l = 0; l < network.depth; l++) {
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
