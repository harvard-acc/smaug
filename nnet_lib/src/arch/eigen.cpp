#include <assert.h>

#include "gem5/m5ops.h"

#include "arch/common.h"
#include "arch/interface.h"
#include "core/eigen/activation_functions.h"
#include "core/eigen/batch_norm.h"
#include "core/eigen/convolution.h"
#include "core/eigen/matrix_multiply.h"
#include "core/eigen/pooling.h"
#include "utility/utility.h"
#include "utility/profiling.h"
#include "nnet_fwd.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

#if ARCHITECTURE == EIGEN
#ifndef __cplusplus
#error "The eigen backend implementation must be built with g++!"
#endif

#include "Eigen/Dense"

// This is an architecture that runs an entire neural network in a single
// block, where nnet_fwd is the top level function. nnet_fwd is thus
// responsible for ensuring that all activations and weights data is available
// when
// each layer needs them.

result_buf inner_product_layer(float* activations,
                               float* weights,
                               layer_t* layers,
                               int lnum,
                               float* result,
                               device_t* device) {
    nnet_eigen::matrix_multiply_with_bias(
            activations, weights, NUM_TEST_CASES, layers[lnum].weights.rows,
            layers[lnum].weights.cols + layers[lnum].weights.align_pad, result);
    return result;
}

result_buf convolution_layer(float* activations,
                             float* kernels,
                             layer_t* layers,
                             int lnum,
                             float* result,
                             device_t* device) {
    layer_t curr_layer = layers[lnum];
    nnet_eigen::convolution3d(activations, kernels, layers + lnum, result);
    return result;
}

result_buf pooling_layer(float* activations,
                         layer_t* layers,
                         int lnum,
                         float* result,
                         device_t* device) {
    layer_t curr_layer = layers[lnum];
    if (curr_layer.pool == MAX) {
        nnet_eigen::max_pooling(activations, result, curr_layer);
    } else {
        assert(false && "Unsupported pooling layer type!");
    }
    return result;
}

result_buf batch_norm_layer(float* activations,
                            float* weights,
                            layer_t* layers,
                            int lnum,
                            float* result,
                            device_t* device) {
    int input_size =
            layers[lnum].inputs.rows *
            (layers[lnum].inputs.cols + layers[lnum].inputs.align_pad) *
            layers[lnum].inputs.height;
    nnet_eigen::batch_norm(activations, weights, input_size, NUM_TEST_CASES, result);
    return result;
}

result_buf activation_sublayer(float* activations,
                               layer_t* layers,
                               int lnum,
                               float* result) {
    int size = get_output_activations_size(&layers[lnum]);
    nnet_eigen::activation_fun(
            activations, size, layers[lnum].activation, sigmoid_table, result);
    return result;
}

result_buf run_layer(float* activations,
                     float* weights,
                     layer_t* layers,
                     int layer_num,
                     float* result,
                     device_t* device) {
    begin_profiling("run_layer", layers + layer_num, layer_num);
    layer_t curr_layer = layers[layer_num];

    begin_profiling(
            "run_layer_skip_activation_func", layers + layer_num, layer_num);
    result_buf result_loc = run_layer_skip_activation_func(
            activations, weights, layers, layer_num, result, device);
    end_profiling();

    if (curr_layer.activation != NO_ACTIVATION) {
        PRINT_MSG("\nactivation function\n");
        begin_profiling("activation_fun", layers + layer_num, layer_num);
        // Pass through activation function
        if (result_loc == activations) {
            result_loc = activation_sublayer(activations, layers, layer_num, result);
        } else {
            result_loc = activation_sublayer(result, layers, layer_num, activations);
        }
        end_profiling();

        PRINT_DEBUG4D(result_loc, curr_layer.outputs.rows,
                      curr_layer.outputs.cols + curr_layer.outputs.align_pad,
                      curr_layer.outputs.height);
    }
    end_profiling();
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

        grab_weights_dma(weights.d, weights.d, l, network.layers);

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
