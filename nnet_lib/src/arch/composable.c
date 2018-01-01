#include <assert.h>

#include "nnet_fwd.h"
#include "core/ref/activation_functions.h"
#include "core/ref/batch_norm.h"
#include "core/ref/convolution.h"
#include "core/ref/flatten.h"
#include "core/ref/matrix_multiply.h"
#include "core/ref/pooling.h"
#include "core/ref/zeropad.h"
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
unsigned kBatchNormHw = 0x0005;

// This is an architecture that divides each layer type into a separate
// hardware block. This is represented by ensuring that each layer is
// responsible for loading its own input activations and weights. For clarity,
// all functions to be turned into hardware are suffixed with _hw.

result_buf flatten_input(float* activations,
                         layer_t* layers,
                         int lnum,
                         float* result) {
    return flatten_input_rowmajor(activations, layers, lnum, result);
}

void inner_product_layer_hw(float* activations,
                            float* weights,
                            layer_t* layers,
                            int lnum,
                            float* result) {
    grab_weights_dma(weights, weights, lnum, layers);
    grab_input_activations_dma(activations, activations, &layers[lnum]);
#if TRANSPOSE_WEIGHTS == 0
    matrix_multiply_with_bias(
            activations, weights, NUM_TEST_CASES, layers[lnum].weights.rows,
            layers[lnum].weights.cols + layers[lnum].weights.align_pad,
            result);
#else
    matrix_multiply_with_bias_transpose(
            activations, weights, NUM_TEST_CASES, layers[lnum].weights.rows,
            layers[lnum].weights.cols + layers[lnum].weights.align_pad,
            result);
#endif
    store_output_activations_dma(result, result, &layers[lnum]);
}

result_buf inner_product_layer(float* activations,
                               float* weights,
                               layer_t* layers,
                               int lnum,
                               float* result,
                               device_t* device) {
    MAP_ARRAY(kInnerProductHw, activations, INPUT_BYTES(layers, lnum));
    MAP_ARRAY(kInnerProductHw, weights, WEIGHT_BYTES(layers, lnum));
    MAP_ARRAY(kInnerProductHw, result, OUTPUT_BYTES(layers, lnum));
    INVOKE_KERNEL(kInnerProductHw, inner_product_layer_hw, activations, weights,
                  layers, lnum, result);
    return result;
}

void standard_convolution_layer_hw(float* activations,
                                   float* weights,
                                   layer_t* layers,
                                   int lnum,
                                   float* result) {
    layer_t curr_layer = layers[lnum];
    grab_weights_dma(weights, weights, lnum, layers);
    grab_input_activations_dma(activations, activations, &layers[lnum]);
    convolution3d_no_padding(activations, weights, curr_layer, result);
    store_output_activations_dma(result, result, &layers[lnum]);
}

void depthwise_convolution_layer_hw(float* activations,
                                    float* weights,
                                    layer_t* layers,
                                    int lnum,
                                    float* result) {
    layer_t curr_layer = layers[lnum];
    grab_weights_dma(weights, weights, lnum, layers);
    grab_input_activations_dma(activations, activations, &layers[lnum]);
    convolution2d_depthwise_nopadding(activations, weights, curr_layer, result);
    store_output_activations_dma(result, result, &layers[lnum]);
}

void pointwise_convolution_layer_hw(float* activations,
                                    float* weights,
                                    layer_t* layers,
                                    int lnum,
                                    float* result) {
    layer_t curr_layer = layers[lnum];
    grab_weights_dma(weights, weights, lnum, layers);
    grab_input_activations_dma(activations, activations, &layers[lnum]);
    convolution3d_pointwise_nopadding(activations, weights, curr_layer, result);
    store_output_activations_dma(result, result, &layers[lnum]);
}

result_buf standard_convolution_layer(float* activations,
                                      float* weights,
                                      layer_t* layers,
                                      int lnum,
                                      float* result,
                                      device_t* device) {
    MAP_ARRAY(kConvolutionHw, activations, INPUT_BYTES(layers, lnum));
    MAP_ARRAY(kConvolutionHw, weights, WEIGHT_BYTES(layers, lnum));
    MAP_ARRAY(kConvolutionHw, result, OUTPUT_BYTES(layers, lnum));

    layer_t curr_layer = layers[lnum];
    if (curr_layer.c_padding > 0) {
        // TODO: Replace this with a memcpy implementation.
        copy_zeropad(activations, layers, lnum, result);
        INVOKE_KERNEL(kConvolutionHw, standard_convolution_layer_hw, result,
                      weights, layers, lnum, activations);

        return activations;
    }
    INVOKE_KERNEL(kConvolutionHw, standard_convolution_layer_hw, activations,
                  weights, layers, lnum, result);
    return result;
}

result_buf depthwise_convolution_layer(float* activations,
                                       float* weights,
                                       layer_t* layers,
                                       int lnum,
                                       float* result,
                                       device_t* device) {
    MAP_ARRAY(kConvolutionHw, activations, INPUT_BYTES(layers, lnum));
    MAP_ARRAY(kConvolutionHw, weights, WEIGHT_BYTES(layers, lnum));
    MAP_ARRAY(kConvolutionHw, result, OUTPUT_BYTES(layers, lnum));

    layer_t curr_layer = layers[lnum];
    if (curr_layer.c_padding > 0) {
        copy_zeropad(activations, layers, lnum, result);
        INVOKE_KERNEL(kConvolutionHw, depthwise_convolution_layer_hw, result,
                      weights, layers, lnum, activations);

        return activations;
    }
    INVOKE_KERNEL(kConvolutionHw, depthwise_convolution_layer_hw, activations,
                  weights, layers, lnum, result);
    return result;
}

result_buf pointwise_convolution_layer(float* activations,
                                       float* weights,
                                       layer_t* layers,
                                       int lnum,
                                       float* result,
                                       device_t* device) {
    MAP_ARRAY(kConvolutionHw, activations, INPUT_BYTES(layers, lnum));
    MAP_ARRAY(kConvolutionHw, weights, WEIGHT_BYTES(layers, lnum));
    MAP_ARRAY(kConvolutionHw, result, OUTPUT_BYTES(layers, lnum));
    INVOKE_KERNEL(kConvolutionHw, pointwise_convolution_layer_hw, activations,
                  weights, layers, lnum, result);
    return result;
}

void max_pooling_layer_hw(float* activations,
                          float* result,
                          layer_t* layers,
                          int lnum) {
    layer_t curr_layer = layers[lnum];
    grab_input_activations_dma(activations, activations, &layers[lnum]);
    max_pooling(activations, result, curr_layer);
    store_output_activations_dma(result, result, &layers[lnum]);
}

result_buf pooling_layer(float* activations,
                         layer_t* layers,
                         int lnum,
                         float* result,
                         device_t* device) {
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

void batch_norm_layer_hw(float* activations,
                          float* weights,
                          layer_t* layers,
                          int lnum,
                          float* result) {
    layer_t curr_layer = layers[lnum];
    grab_weights_dma(weights, weights, lnum, layers);
    grab_input_activations_dma(activations, activations, &layers[lnum]);
    batch_norm_fxp(activations, weights, &layers[lnum], NUM_TEST_CASES, result);
    store_output_activations_dma(result, result, &layers[lnum]);
}

result_buf batch_norm_layer(float* activations,
                            float* weights,
                            layer_t* layers,
                            int lnum,
                            float* result,
                            device_t* device) {
    MAP_ARRAY(kBatchNormHw, activations, INPUT_BYTES(layers, lnum));
    MAP_ARRAY(kBatchNormHw, weights, WEIGHT_BYTES(layers, lnum));
    MAP_ARRAY(kBatchNormHw, result, OUTPUT_BYTES(layers, lnum));

    INVOKE_KERNEL(kBatchNormHw, batch_norm_layer_hw, activations,
                  weights, layers, lnum, result);

    return result;
}

void activation_hw(float* activations,
                   layer_t* layers,
                   int lnum,
                   float* sigmoid_table) {
    layer_t curr_layer = layers[lnum];
    int input_size = grab_output_activations_dma(
                             activations, activations, &layers[lnum]) /
                     NUM_TEST_CASES;
    activation_fun(activations, NUM_TEST_CASES, input_size,
                   curr_layer.activation, sigmoid_table);
    store_output_activations_dma(activations, activations, &layers[lnum]);
}

result_buf activation_sublayer(float* activations,
                               layer_t* layers,
                               int lnum) {
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
                     device_t* device) {
    layer_t curr_layer = layers[layer_num];

    result_buf result_loc = run_layer_skip_activation_func(
            activations, weights, layers, layer_num, result, device);

    if (curr_layer.activation != NO_ACTIVATION) {
        PRINT_MSG("\nactivation function\n");
        // Pass through activation function
        if (result_loc == activations) {
            activation_sublayer(activations, layers, layer_num);
        } else {
            activation_sublayer(result, layers, layer_num);
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
              device_t* device) {

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
                                   activations.d, device);
        } else {
            result_loc = run_layer(activations.d, weights.d, network.layers, l,
                                   result.d, device);
        }
    }

    network.layers[network.depth - 1].result_in_temp = (result_loc == result.d);

    if (result_loc == result.d)
        dmaStore(result.d, result.d,
                 NUM_TEST_CASES * NUM_CLASSES * sizeof(float));
    else
        dmaStore(activations.d, activations.d,
                 NUM_TEST_CASES * NUM_CLASSES * sizeof(float));
    dmaStore(network.layers, network.layers, network.depth * sizeof(layer_t));
}

#endif
