#include <assert.h>

#include "nnet_fwd.h"
#include "arch/common.h"
#include "arch/interface.h"
#include "core/ref/activation_functions.h"
#include "core/ref/batch_norm.h"
#include "core/ref/convolution.h"
#include "core/ref/matrix_multiply.h"
#include "core/ref/pooling.h"
#include "core/ref/zeropad.h"
#include "utility/data_layout_conversion.h"
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

result_buf flatten_input(data_list* activations,
                         layer_t* layers,
                         int lnum,
                         data_list* results) {
    require_data_type(activations, 0, Uncompressed);
    return im2row(activations, layers, lnum, results);
}

result_buf inner_product_layer(data_list* activations,
                               data_list* weights,
                               layer_t* layers,
                               int lnum,
                               data_list* results,
                               device_t* device,
                               sampling_param_t* sampling_param) {
    require_data_type(activations, 0, Uncompressed);
    require_data_type(weights, 0, Uncompressed);
    results = create_new_data_list_if_necessary(
            results,
            NUM_TEST_CASES * get_dims_size(&layers[lnum].outputs),
            Uncompressed);
    // These kernels fuse the bias with the GEMM and assume that the rows
    // parameter includes the extra row of biases.
#if TRANSPOSE_WEIGHTS == 0
    matrix_multiply_with_bias(
            activations->data[0].dense->d, weights->data[0].dense->d,
            NUM_TEST_CASES, layers[lnum].weights.rows + 1,
            layers[lnum].weights.cols + layers[lnum].weights.align_pad,
            results->data[0].dense->d);
#else
    matrix_multiply_with_bias_transpose(
            activations->data[0].dense->d, weights->data[0].dense->d,
            NUM_TEST_CASES, layers[lnum].weights.rows + 1,
            layers[lnum].weights.cols + layers[lnum].weights.align_pad,
            results->data[0].dense->d);
#endif
    return results;
}

result_buf standard_convolution_layer(data_list* activations,
                                      data_list* kernels,
                                      layer_t* layers,
                                      int lnum,
                                      data_list* results,
                                      device_t* device,
                                      sampling_param_t* sampling_param) {
    require_data_type(activations, 0, Uncompressed);
    require_data_type(kernels, 0, Uncompressed);
    if (layers[lnum].c_padding > 0) {
        // TODO: Get rid of all this manual zeropadding!
        //
        // @results will contain the zeropadded image. The convolution routine
        // will then process that buffer and produce its outputs in
        // @activations.
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
    convolution3d_no_padding(activations->data[0].dense->d,
                             kernels->data[0].dense->d, layers[lnum],
                             results->data[0].dense->d);
    return results;
}

result_buf depthwise_convolution_layer(data_list* activations,
                                       data_list* kernels,
                                       layer_t* layers,
                                       int lnum,
                                       data_list* results,
                                       device_t* device,
                                       sampling_param_t* sampling_param) {
    require_data_type(activations, 0, Uncompressed);
    require_data_type(kernels, 0, Uncompressed);
    if (layers[lnum].c_padding > 0) {
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
    convolution2d_depthwise_nopadding(activations->data[0].dense->d,
                                      kernels->data[0].dense->d, layers[lnum],
                                      results->data[0].dense->d);
    return results;
}

result_buf pointwise_convolution_layer(data_list* activations,
                                       data_list* kernels,
                                       layer_t* layers,
                                       int lnum,
                                       data_list* results,
                                       device_t* device,
                                       sampling_param_t* sampling_param) {
    require_data_type(activations, 0, Uncompressed);
    require_data_type(kernels, 0, Uncompressed);
    results = create_new_data_list_if_necessary(
            results,
            NUM_TEST_CASES * get_dims_size(&layers[lnum].outputs),
            Uncompressed);
    convolution3d_pointwise_nopadding(activations->data[0].dense->d,
                                      kernels->data[0].dense->d, layers[lnum],
                                      results->data[0].dense->d);
    return results;
}

result_buf pooling_layer(data_list* activations,
                         layer_t* layers,
                         int lnum,
                         data_list* results,
                         device_t* device,
                         sampling_param_t* sampling_param) {
    require_data_type(activations, 0, Uncompressed);
    results = create_new_data_list_if_necessary(
            results,
            NUM_TEST_CASES * get_dims_size(&layers[lnum].outputs),
            Uncompressed);
    layer_t curr_layer = layers[lnum];
    if (curr_layer.pool == MAX) {
        max_pooling(activations->data[0].dense->d, results->data[0].dense->d,
                    curr_layer);
    } else if (curr_layer.pool == AVG) {
        avg_pooling(activations->data[0].dense->d, results->data[0].dense->d,
                    curr_layer);
    } else {
        assert(false && "Unsupported pooling layer type!");
    }
    return results;
}

result_buf batch_norm_layer(data_list* activations,
                            data_list* weights,
                            layer_t* layers,
                            int lnum,
                            data_list* results,
                            device_t* device,
                            sampling_param_t* sampling_param) {
    require_data_type(activations, 0, Uncompressed);
    require_data_type(weights, 0, Uncompressed);
    results = create_new_data_list_if_necessary(
            results,
            NUM_TEST_CASES * get_dims_size(&layers[lnum].outputs),
            Uncompressed);
    batch_norm_fxp(activations->data[0].dense->d, weights->data[0].dense->d,
                   &layers[lnum], NUM_TEST_CASES, results->data[0].dense->d);
    return results;
}

result_buf activation_sublayer(data_list* activations,
                               layer_t* layers,
                               int lnum) {
    require_data_type(activations, 0, Uncompressed);
    int input_size = get_dims_size(&layers[lnum].outputs);
    activation_fun(activations->data[0].dense->d, NUM_TEST_CASES, input_size,
                   layers[lnum].activation);
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

    // FORMAT HERE IS H TIMES W, NOT W TIMES H!!!!!
    // SO EACH DATA POINT IS A ***ROW****

    //******************//
    //   PRIMARY LOOP   //
    //******************//

    nnet_fwd_outer:
    for (int l = 1; l < network->depth; l++) {
        curr_layer = network->layers[l];

        if (result_loc == results) {
            result_loc = run_layer(results,
                                   curr_layer.host_weights,
                                   network->layers,
                                   l,
                                   activations,
                                   device,
                                   sampling_param);
        } else {
            result_loc = run_layer(activations,
                                   curr_layer.host_weights,
                                   network->layers,
                                   l,
                                   results,
                                   device,
                                   sampling_param);
        }
    }

    network->layers[network->depth - 1].result_in_temp =
            (result_loc == results);
}

#endif
