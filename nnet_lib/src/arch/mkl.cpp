#include <iostream>
#include <list>
#include <cassert>

#include "arch/common.h"
#include "arch/interface.h"
#include "arch/nnet_mkl.h"
#include "core/mkl/activation_functions.h"
#include "core/mkl/batch_norm.h"
#include "core/mkl/convolution.h"
#include "core/mkl/matrix_multiply.h"
#include "core/mkl/pooling.h"
#include "utility/data_layout_conversion.h"
#include "utility/utility.h"
#include "utility/mkl/utility.h"

#include "nnet_fwd.h"

#include "mkldnn.hpp"

#if ARCHITECTURE == MKLDNN

/* Memory management in the MKL backend.
 *
 * The MKL backend has a complicated memory management system. There are two
 * ways to manage memory for the result of MKL operations.
 *
 * 1. If we pass a non-NULL pointer for the output buffers, it will use that
 *    buffer, and we have to manage the lifetime of that buffer.
 * 2. If we pass a NULL pointer, then MKL will allocate storage and manage it
 *    automatically.
 *
 * The first option worked fine when there were only two buffers we had to
 * manage. Now that we allow each layer to create its own output memory buffer,
 * we let MKL do the memory management on its own, which is why all the calls
 * to the MKL wrapper functions pass NULL as the output buffer pointer.
 *
 * However, after constructing an MKL primitive, we still need to retrieve a
 * data list corresponding to its output, because it will be the input to the
 * next layer. We generally don't actually USE it, because the MKL wrappers can
 * automatically reuse the output memory primitive from the last operation in
 * the MKL session, but we still need to fetch a data list, as the SMAUG API
 * expects it. This is an ugly part of the API that should be addressed.
 */

// Stores the data_list objects containing the intermediate results of each layer.
static std::list<data_list*> intermediate_results;

// Returns a data_list object containing the results of the last operation in
// the MklSession. It is added to the list of intermediate results, so that
// they can be freed when the network finishes.
data_list* get_result_data_list(device_t* device) {
    data_list* results =
            nnet_mkl::get_session(device)->last_op()->get_output_data_list();
    intermediate_results.push_back(results);
    return results;
}

// Free the memory for the data list/array containers, but NOT the underlying
// buffers that they describe!
void free_intermediate_results() {
    for (auto data_list : intermediate_results) {
        for (int i = 0; i < data_list->len; i++) {
            data_storage_t type = data_list->type[i];
            if (type == Uncompressed) {
                farray_t* array = data_list->data[i].dense;
                free(array);
            } else if (type == UncompressedHalfPrecision) {
                uarray_t* array = data_list->data[i].dense_hp;
                free(array);
            } else {
                std::cerr << "[WARNING]: Tried to free intermediate results in "
                             "format \""
                          << data_storage_str(type)
                          << "\", which is not supported by MKL!\n";
            }
        }
        free(data_list->data);
        free(data_list->type);
        free(data_list);
    }
    intermediate_results.clear();
}

// MKL does not need us to explicitly flatten the input, since it has
// reordering primitives that will handle it.
result_buf flatten_input(data_list* activations,
                         layer_t* layers,
                         int lnum,
                         data_list* results) {
    return activations;
}

result_buf inner_product_layer(data_list* activations,
                               data_list* weights,
                               layer_t* layers,
                               int lnum,
                               data_list* results,
                               device_t* device,
                               sampling_param_t* sampling_param) {
    layer_t* curr_layer = &layers[lnum];
    PRINT_DEBUG(weights->data[0].dense->d, curr_layer->weights.rows,
                curr_layer->weights.cols, curr_layer->weights.cols);
    nnet_mkl::matrix_multiply_with_bias(activations->data[0].dense->d,
                                        weights->data[0].dense->d, curr_layer,
                                        NULL, device);
    return get_result_data_list(device);
}

result_buf standard_convolution_layer(data_list* activations,
                                      data_list* weights,
                                      layer_t* layers,
                                      int lnum,
                                      data_list* results,
                                      device_t* device,
                                      sampling_param_t* sampling_param) {
    layer_t* curr_layer = &layers[lnum];
    nnet_mkl::convolution3d(activations->data[0].dense->d,
                            weights->data[0].dense->d, curr_layer,
                            NULL, device);
    return get_result_data_list(device);
}

result_buf depthwise_convolution_layer(data_list* activations,
                                       data_list* weights,
                                       layer_t* layers,
                                       int lnum,
                                       data_list* results,
                                       device_t* device,
                                       sampling_param_t* sampling_param) {
    layer_t* curr_layer = &layers[lnum];
    nnet_mkl::depthwise_convolution3d(activations->data[0].dense->d,
                                      weights->data[0].dense->d, curr_layer,
                                      NULL, device);
    return get_result_data_list(device);
}

result_buf pointwise_convolution_layer(data_list* activations,
                                       data_list* weights,
                                       layer_t* layers,
                                       int lnum,
                                       data_list* results,
                                       device_t* device,
                                       sampling_param_t* sampling_param) {
    layer_t* curr_layer = &layers[lnum];
    nnet_mkl::pointwise_convolution3d(activations->data[0].dense->d,
                                      weights->data[0].dense->d, curr_layer,
                                      NULL, device);
    return get_result_data_list(device);
}

result_buf pooling_layer(data_list* activations,
                         layer_t* layers,
                         int lnum,
                         data_list* results,
                         device_t* device,
                         sampling_param_t* sampling_param) {
    layer_t* curr_layer = &layers[lnum];
    if (layers[lnum].pool == MAX) {
        nnet_mkl::max_pooling_3d(activations->data[0].dense->d, curr_layer,
                                 NULL, device);
    } else if (layers[lnum].pool == AVG) {
        nnet_mkl::avg_pooling_3d(activations->data[0].dense->d, curr_layer,
                                 NULL, device);
    }
    return get_result_data_list(device);
}

result_buf batch_norm_layer(data_list* activations,
                            data_list* weights,
                            layer_t* layers,
                            int lnum,
                            data_list* results,
                            device_t* device,
                            sampling_param_t* sampling_param) {
    layer_t* curr_layer = &layers[lnum];
    nnet_mkl::batch_norm(activations->data[0].dense->d,
                         weights->data[0].dense->d, curr_layer,
                         NUM_TEST_CASES, NULL, device);
    return get_result_data_list(device);
}

result_buf activation_sublayer(data_list* activations,
                               layer_t* layers,
                               int lnum,
                               data_list* results,
                               device_t* device) {
    layer_t* curr_layer = &layers[lnum];
    nnet_mkl::activation_fun(activations->data[0].dense->d, NUM_TEST_CASES,
                             curr_layer, NULL, device);
    return get_result_data_list(device);
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
                                             NULL,
                                             device,
                                             sampling_param);

    if (curr_layer.activation != NO_ACTIVATION) {
        PRINT_MSG("\nactivation function\n");
        result_loc = activation_sublayer(
                result_loc, layers, layer_num, NULL, device);

        PRINT_DEBUG4D(result_loc->data[0].dense->d, curr_layer.outputs.rows,
                      curr_layer.outputs.cols + curr_layer.outputs.align_pad,
                      curr_layer.outputs.height);
    }
    return result_loc;
}

void nnet_fwd(data_list* activations,
              data_list* weights,
              data_list* results,
              network_t* network,
              device_t* device,
              sampling_param_t* sampling_param) {
    layer_t* layers = network->layers;
    nnet_mkl::MklSession* session = new nnet_mkl::MklSession();
    device->session = (void*)session;

    // Alternate between reading from/writing to activations and results so we
    // can avoid copying matrices. The initial activations is obviously in
    // "activations", so that's where we start.
    result_buf result_loc = activations;
    // Allocate an empty farray_t to hold a null pointer (otherwise we'd
    // segfault when we tried to access farray->d).
    results->data[0].dense = init_farray(0, false);

    // This only creates all the MKL primitives that comprise the network and
    // the memory buffers to store inputs, weights, and results.
    nnet_fwd_outer:
    for (int l = 1; l < network->depth; l++) {
        if (result_loc == results)
            SWAP_PTRS(activations, results);
        result_loc = run_layer(activations, layers[l].host_weights, layers, l,
                               results, device, sampling_param);
    }

    // Now run the network.
    session->run();

    // Copy the final output into the results buffer.
    results = copy_data_list(results, result_loc);
    layers[network->depth - 1].result_in_temp = true;
    free_intermediate_results();
    delete session;
    device->session = NULL;
}

#endif
