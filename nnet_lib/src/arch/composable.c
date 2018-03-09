#include <assert.h>

#include "nnet_fwd.h"
#include "core/ref/activation_functions.h"
#include "core/ref/batch_norm.h"
#include "core/ref/convolution.h"
#include "core/ref/matrix_multiply.h"
#include "core/ref/pooling.h"
#include "core/ref/zeropad.h"
#include "utility/data_layout_conversion.h"
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

result_buf flatten_input(data_list* activations,
                         layer_t* layers,
                         int lnum,
                         data_list* results) {
    require_data_type(activations, 0, Uncompressed);
    return im2row(activations, layers, lnum, results);
}

void inner_product_layer_hw(float* activations,
                            float* weights,
                            layer_t* layers,
                            int lnum,
                            float* results) {
    grab_input_activations_dma(activations, activations, &layers[lnum]);
#if TRANSPOSE_WEIGHTS == 0
    matrix_multiply_with_bias(
            activations, weights, NUM_TEST_CASES, layers[lnum].weights.rows + 1,
            layers[lnum].weights.cols + layers[lnum].weights.align_pad, results);
#else
    matrix_multiply_with_bias_transpose(
            activations, weights, NUM_TEST_CASES, layers[lnum].weights.rows + 1,
            layers[lnum].weights.cols + layers[lnum].weights.align_pad, results);
#endif
    store_output_activations_dma(results, results, &layers[lnum]);
}

result_buf inner_product_layer(data_list* activations,
                               data_list* weights,
                               layer_t* layers,
                               int lnum,
                               data_list* results,
                               device_t* device,
                               sampling_param_t* sampling_param) {
    results = create_new_data_list_if_necessary(
            results,
            NUM_TEST_CASES * get_dims_size(&layers[lnum].outputs),
            Uncompressed);
    float* act_buf = activations->data[0].dense->d;
    float* wgt_buf = weights->data[0].dense->d;
    float* out_buf = results->data[0].dense->d;
    MAP_ARRAY(kInnerProductHw, act_buf, INPUT_BYTES(layers, lnum));
    MAP_ARRAY(kInnerProductHw, wgt_buf, WEIGHT_BYTES(layers, lnum));
    MAP_ARRAY(kInnerProductHw, out_buf, OUTPUT_BYTES(layers, lnum));
    INVOKE_KERNEL(kInnerProductHw, inner_product_layer_hw, act_buf, wgt_buf,
                  layers, lnum, out_buf);
    return results;
}

void standard_convolution_layer_hw(float* activations,
                                   float* weights,
                                   layer_t* layers,
                                   int lnum,
                                   float* results) {
    layer_t curr_layer = layers[lnum];
    grab_input_activations_dma(activations, activations, &layers[lnum]);
    convolution3d_no_padding(activations, weights, curr_layer, results);
    store_output_activations_dma(results, results, &layers[lnum]);
}

void depthwise_convolution_layer_hw(float* activations,
                                    float* weights,
                                    layer_t* layers,
                                    int lnum,
                                    float* results) {
    layer_t curr_layer = layers[lnum];
    grab_input_activations_dma(activations, activations, &layers[lnum]);
    convolution2d_depthwise_nopadding(activations, weights, curr_layer, results);
    store_output_activations_dma(results, results, &layers[lnum]);
}

void pointwise_convolution_layer_hw(float* activations,
                                    float* weights,
                                    layer_t* layers,
                                    int lnum,
                                    float* results) {
    layer_t curr_layer = layers[lnum];
    grab_input_activations_dma(activations, activations, &layers[lnum]);
    convolution3d_pointwise_nopadding(activations, weights, curr_layer, results);
    store_output_activations_dma(results, results, &layers[lnum]);
}

result_buf standard_convolution_layer(data_list* activations,
                                      data_list* weights,
                                      layer_t* layers,
                                      int lnum,
                                      data_list* results,
                                      device_t* device,
                                      sampling_param_t* sampling_param) {
    layer_t curr_layer = layers[lnum];
    if (curr_layer.c_padding > 0) {
        results = create_new_data_list_if_necessary(
                results,
                NUM_TEST_CASES * get_dims_size(&layers[lnum].inputs),
                Uncompressed);
        copy_zeropad(activations->data[0].dense->d, layers, lnum,
                     results->data[0].dense->d);
        SWAP_PTRS(results, activations);
    }

    results = create_new_data_list_if_necessary(
            results,
            NUM_TEST_CASES * get_dims_size(&layers[lnum].outputs),
            Uncompressed);
    float* act_buf = activations->data[0].dense->d;
    float* wgt_buf = weights->data[0].dense->d;
    float* out_buf = results->data[0].dense->d;
    MAP_ARRAY(kConvolutionHw, act_buf, INPUT_BYTES(layers, lnum));
    MAP_ARRAY(kConvolutionHw, wgt_buf, WEIGHT_BYTES(layers, lnum));
    MAP_ARRAY(kConvolutionHw, out_buf, OUTPUT_BYTES(layers, lnum));

    INVOKE_KERNEL(kConvolutionHw, standard_convolution_layer_hw, act_buf,
                  wgt_buf, layers, lnum, out_buf);
    return results;
}

result_buf depthwise_convolution_layer(data_list* activations,
                                       data_list* weights,
                                       layer_t* layers,
                                       int lnum,
                                       data_list* results,
                                       device_t* device,
                                       sampling_param_t* sampling_param) {
    layer_t curr_layer = layers[lnum];
    if (curr_layer.c_padding > 0) {
        results = create_new_data_list_if_necessary(
                results,
                NUM_TEST_CASES * get_dims_size(&layers[lnum].inputs),
                Uncompressed);
        copy_zeropad(activations->data[0].dense->d, layers, lnum,
                     results->data[0].dense->d);
        SWAP_PTRS(results, activations);
    }

    results = create_new_data_list_if_necessary(
            results,
            NUM_TEST_CASES * get_dims_size(&layers[lnum].outputs),
            Uncompressed);
    float* act_buf = activations->data[0].dense->d;
    float* wgt_buf = weights->data[0].dense->d;
    float* out_buf = results->data[0].dense->d;
    MAP_ARRAY(kConvolutionHw, act_buf, INPUT_BYTES(layers, lnum));
    MAP_ARRAY(kConvolutionHw, wgt_buf, WEIGHT_BYTES(layers, lnum));
    MAP_ARRAY(kConvolutionHw, out_buf, OUTPUT_BYTES(layers, lnum));

    INVOKE_KERNEL(kConvolutionHw, depthwise_convolution_layer_hw, act_buf,
                  wgt_buf, layers, lnum, out_buf);
    return results;
}

result_buf pointwise_convolution_layer(data_list* activations,
                                       data_list* weights,
                                       layer_t* layers,
                                       int lnum,
                                       data_list* results,
                                       device_t* device,
                                       sampling_param_t* sampling_param) {
    results = create_new_data_list_if_necessary(
            results,
            NUM_TEST_CASES * get_dims_size(&layers[lnum].outputs),
            Uncompressed);
    float* act_buf = activations->data[0].dense->d;
    float* wgt_buf = weights->data[0].dense->d;
    float* out_buf = results->data[0].dense->d;
    MAP_ARRAY(kConvolutionHw, act_buf, INPUT_BYTES(layers, lnum));
    MAP_ARRAY(kConvolutionHw, wgt_buf, WEIGHT_BYTES(layers, lnum));
    MAP_ARRAY(kConvolutionHw, out_buf, OUTPUT_BYTES(layers, lnum));
    INVOKE_KERNEL(kConvolutionHw, pointwise_convolution_layer_hw, act_buf,
                  wgt_buf, layers, lnum, out_buf);
    return results;
}

void max_pooling_layer_hw(float* activations,
                          float* results,
                          layer_t* layers,
                          int lnum) {
    layer_t curr_layer = layers[lnum];
    grab_input_activations_dma(activations, activations, &layers[lnum]);
    max_pooling(activations, results, curr_layer);
    store_output_activations_dma(results, results, &layers[lnum]);
}

void avg_pooling_layer_hw(float* activations,
                          float* results,
                          layer_t* layers,
                          int lnum) {
    layer_t curr_layer = layers[lnum];
    grab_input_activations_dma(activations, activations, &layers[lnum]);
    avg_pooling(activations, results, curr_layer);
    store_output_activations_dma(results, results, &layers[lnum]);
}

result_buf pooling_layer(data_list* activations,
                         layer_t* layers,
                         int lnum,
                         data_list* results,
                         device_t* device,
                         sampling_param_t* sampling_param) {
    layer_t curr_layer = layers[lnum];
    results = create_new_data_list_if_necessary(
            results,
            NUM_TEST_CASES * get_dims_size(&curr_layer.outputs),
            Uncompressed);
    float* act_buf = activations->data[0].dense->d;
    float* out_buf = results->data[0].dense->d;
    MAP_ARRAY(kPoolingHw, act_buf, INPUT_BYTES(layers, lnum));
    MAP_ARRAY(kPoolingHw, out_buf, OUTPUT_BYTES(layers, lnum));
    if (curr_layer.pool == MAX) {
        INVOKE_KERNEL(kPoolingHw, max_pooling_layer_hw, act_buf, out_buf,
                      layers, lnum);
    } else if (curr_layer.pool == AVG) {
        INVOKE_KERNEL(kPoolingHw, avg_pooling_layer_hw, act_buf, out_buf,
                      layers, lnum);
    } else {
        assert(false && "Unsupported pooling layer type!");
    }
    return results;
}

void batch_norm_layer_hw(float* activations,
                         float* weights,
                         layer_t* layers,
                         int lnum,
                         float* results) {
    grab_input_activations_dma(activations, activations, &layers[lnum]);
    batch_norm_fxp(activations, weights, &layers[lnum], NUM_TEST_CASES, results);
    store_output_activations_dma(results, results, &layers[lnum]);
}

result_buf batch_norm_layer(data_list* activations,
                            data_list* weights,
                            layer_t* layers,
                            int lnum,
                            data_list* results,
                            device_t* device,
                            sampling_param_t* sampling_param) {
    results = create_new_data_list_if_necessary(
            results,
            NUM_TEST_CASES * get_dims_size(&layers[lnum].outputs),
            Uncompressed);
    float* act_buf = activations->data[0].dense->d;
    float* wgt_buf = weights->data[0].dense->d;
    float* out_buf = results->data[0].dense->d;
    MAP_ARRAY(kBatchNormHw, act_buf, INPUT_BYTES(layers, lnum));
    MAP_ARRAY(kBatchNormHw, wgt_buf, WEIGHT_BYTES(layers, lnum));
    MAP_ARRAY(kBatchNormHw, out_buf, OUTPUT_BYTES(layers, lnum));
    INVOKE_KERNEL(kBatchNormHw, batch_norm_layer_hw, act_buf, wgt_buf, layers,
                  lnum, out_buf);
    return results;
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
                   layers[lnum].outputs.align_pad, curr_layer.activation);
    store_output_activations_dma(activations, activations, &layers[lnum]);
}

result_buf activation_sublayer(data_list* activations,
                               layer_t* layers,
                               int lnum) {
    float* act_buf = activations->data[0].dense->d;
    MAP_ARRAY(kActivationFuncHw, act_buf, OUTPUT_BYTES(layers, lnum));
    INVOKE_KERNEL(kActivationFuncHw, activation_hw, act_buf, layers, lnum,
                  sigmoid_table);
    return activations;
}

result_buf run_layer(data_list* activations,
                     data_list* weights,
                     layer_t* layers,
                     int layer_num,
                     data_list* results,
                     device_t* device,
                     sampling_param_t* sampling_param) {
    layer_t curr_layer = layers[layer_num];

    result_buf result_loc = layer_dispatcher(activations,
                                             weights,
                                             layers,
                                             layer_num,
                                             results,
                                             device,
                                             sampling_param);

    if (curr_layer.activation != NO_ACTIVATION) {
        PRINT_MSG("\nactivation function\n");
        // Pass through activation function
        if (result_loc == activations) {
            activation_sublayer(activations, layers, layer_num);
        } else {
            activation_sublayer(results, layers, layer_num);
        }

        PRINT_DEBUG4D(result_loc->data[0].dense->d, curr_layer.outputs.rows,
                      curr_layer.outputs.cols + curr_layer.outputs.align_pad,
                      curr_layer.outputs.height);
    }

    return result_loc;
}

// Runs the forward pass of a neural network.
//
// This version loads weights on a per layer basis, and activations are
// ping-ponged between two buffers, activations and results.
void nnet_fwd(data_list* activations,
              data_list* weights,
              data_list* results,
              network_t* network,
              device_t* device,
              sampling_param_t* sampling_param) {
    layer_t curr_layer;

    // Alternate between reading from/writing to activations and results so we
    // can avoid copying matrices. The initial activations is obviously in
    // "activations", so that's where we start.
    result_buf result_loc = activations;

    //******************//
    //   PRIMARY LOOP   //
    //******************//

nnet_fwd_outer:
    for (int l = 1; l < network->depth; l++) {
        curr_layer = network->layers[l];

        if (result_loc == results) {
            result_loc =
                    run_layer(results, curr_layer.host_weights, network->layers,
                              l, activations, device, sampling_param);
        } else {
            result_loc = run_layer(activations, curr_layer.host_weights,
                                   network->layers, l, results, device,
                                   sampling_param);
        }
    }

    network->layers[network->depth - 1].result_in_temp = (result_loc == results);
}

#endif
