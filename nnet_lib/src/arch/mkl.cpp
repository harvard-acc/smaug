#include <assert.h>

#include "arch/common.h"
#include "arch/interface.h"
#include "arch/nnet_mkl.h"
#include "core/flatten.h"
#include "core/mkl/activation_functions.h"
#include "core/mkl/batch_norm.h"
#include "core/mkl/convolution.h"
#include "core/mkl/matrix_multiply.h"
#include "core/mkl/pooling.h"
#include "utility/utility.h"

#include "nnet_fwd.h"

#include "mkldnn.hpp"

#if ARCHITECTURE == MKLDNN

nnet_mkl::MklSession* nnet_mkl::get_session(device_t* device) {
    return reinterpret_cast<nnet_mkl::MklSession*>(device->session);
}

result_buf flatten_input(float* activations,
                         layer_t* layers,
                         int lnum,
                         float* result) {
    return flatten_input_rowmajor(activations, layers, lnum, result);
}

result_buf inner_product_layer(float* activations,
                               float* weights,
                               layer_t* layers,
                               int lnum,
                               float* result,
                               device_t* device) {
    float* curr_layer_weights =
            weights + get_weights_loc_for_layer(layers, lnum);
    PRINT_DEBUG(weights, layers[lnum].weights.rows, layers[lnum].weights.cols,
                layers[lnum].weights.cols);
    nnet_mkl::matrix_multiply_with_bias(
            activations, curr_layer_weights, &layers[lnum], result, device);
    return result;
}

result_buf convolution_layer(float* activations,
                             float* weights,
                             layer_t* layers,
                             int lnum,
                             float* result,
                             device_t* device) {
    float* curr_layer_weights =
            weights + get_weights_loc_for_layer(layers, lnum);
    nnet_mkl::convolution3d(
            activations, curr_layer_weights, &layers[lnum], result, device);
    return result;
}

result_buf pooling_layer(float* activations,
                         layer_t* layers,
                         int lnum,
                         float* result,
                         device_t* device) {
    nnet_mkl::max_pooling_3d(activations, &layers[lnum], result, device);
    return result;
}

result_buf batch_norm_layer(float* activations,
                            float* weights,
                            layer_t* layers,
                            int lnum,
                            float* result,
                            device_t* device) {
    float* curr_layer_weights =
            weights + get_weights_loc_for_layer(layers, lnum);
    nnet_mkl::batch_norm(activations, curr_layer_weights, &layers[lnum],
                         NUM_TEST_CASES, result, device);
    return result;
}

result_buf activation_sublayer(float* activations,
                               layer_t* layers,
                               int lnum,
                               float* result,
                               device_t* device) {
    int size = get_output_activations_size(&layers[lnum]) / NUM_TEST_CASES;
    nnet_mkl::activation_fun(activations, NUM_TEST_CASES, size,
                             layers[lnum].activation, result, device);
    return result;
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
            activation_sublayer(activations, layers, layer_num, result, device);
        } else {
            activation_sublayer(result, layers, layer_num, activations, device);
            result_loc = activations;
        }

        PRINT_DEBUG4D(result_loc, curr_layer.outputs.rows,
                      curr_layer.outputs.cols + curr_layer.outputs.align_pad,
                      curr_layer.outputs.height);
    }
    return result_loc;
}

void nnet_fwd(farray_t activations,
              farray_t weights,
              farray_t result,
              network_t network,
              device_t* device) {
    int l;
    layer_t* layers = network.layers;
    nnet_mkl::MklSession* session = new nnet_mkl::MklSession();
    device->session = (void*)session;

    // Alternate between reading from/writing to activations and result so we
    // can
    // avoid copying matrices. The initial activations is obviously in
    // "activations",
    // so that's where we start.
    result_buf result_loc = activations.d;

    // FORMAT HERE IS H TIMES W, NOT W TIMES H!!!!!
    // SO EACH DATA POINT IS A ***ROW****

    l = 0;
    layer_t curr_layer;

    //******************//
    //   PRIMARY LOOP   //
    //******************//

nnet_fwd_outer:
    for (l = 1; l < network.depth; l++) {
        curr_layer = layers[l];

        // grab_weights_dma(weights, weights, l, layers);

        if (result_loc == result.d) {
            result_loc = run_layer(
                    result.d, weights.d, layers, l, activations.d, device);
        } else {
            result_loc = run_layer(
                    activations.d, weights.d, layers, l, result.d, device);
        }
    }

    session->run();

    layers[network.depth - 1].result_in_temp = result_loc == result.d;

    delete session;
}

#endif
