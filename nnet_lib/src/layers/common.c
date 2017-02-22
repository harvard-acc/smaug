#include "nnet_fwd.h"
#include "core/activation_functions.h"
#include "core/convolution.h"
#include "core/matrix_multiply.h"
#include "core/pooling.h"
#include "layers/common.h"
#include "layers/interface.h"
#include "utility/utility.h"

// Common dispatch function for executing a layer.
//
// "Layer" includes the activation function.
//
// A result_buf value is returned indicating the final location of the output of
// this layer. It is either equal to the pointers @activations or @result.
result_buf run_layer(float* activations,
                     float* weights,
                     layer_t* layers,
                     int layer_num,
                     float* result,
                     float* sigmoid_table,
                     bool do_activation_func) {
    layer_t curr_layer = layers[layer_num];
    layer_type l_type = curr_layer.type;
    result_buf result_loc = result;

    if (l_type == FC) {
        PRINT_MSG("\nInner product.\n");
        result_loc = inner_product_layer(
                activations, weights, layers, layer_num, result);
    } else if (l_type == CONV) {
        PRINT_MSG("\nConvolution.\n");
        result_loc = convolution_layer(
                activations, weights, layers, layer_num, result);
    } else if (l_type == POOL_MAX) {
        PRINT_MSG("\nmax pooling\n");
        result_loc = max_pooling_layer(
                activations, layers, layer_num, result);
    }

    if (result_loc == activations) {
        PRINT_DEBUG4D(activations, curr_layer.output_rows,
                      curr_layer.output_cols, curr_layer.output_height);
    } else {
        PRINT_DEBUG4D(result, curr_layer.output_rows,
                      curr_layer.output_cols, curr_layer.output_height);
    }

    if (do_activation_func) {
        PRINT_MSG("\nactivation function\n");
        // Pass through activation function
        if (result_loc == activations) {
            activation_fun(activations,
                           curr_layer.output_rows * curr_layer.output_cols *
                                   curr_layer.output_height,
                           sigmoid_table);
        } else {
            activation_fun(result,
                           curr_layer.output_rows * curr_layer.output_cols *
                                   curr_layer.output_height,
                           sigmoid_table);
        }
    }

    if (result_loc == activations) {
        PRINT_DEBUG4D(activations, curr_layer.output_rows,
                      curr_layer.output_cols, curr_layer.output_height);
    } else {
        PRINT_DEBUG4D(result, curr_layer.output_rows,
                      curr_layer.output_cols, curr_layer.output_height);
    }

    return result_loc;
}
