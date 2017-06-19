#include "nnet_fwd.h"
#include "core/activation_functions.h"
#include "core/convolution.h"
#include "core/matrix_multiply.h"
#include "core/pooling.h"
#include "arch/common.h"
#include "arch/interface.h"
#include "utility/utility.h"

// Common dispatch function for executing a layer.
//
// The activation function is NOT applied on the output.
//
// A result_buf value is returned indicating the final location of the output of
// this layer. It is either equal to the pointers @activations or @result.
result_buf run_layer_skip_activation_func(float* activations,
                                          float* weights,
                                          layer_t* layers,
                                          int layer_num,
                                          float* result,
                                          float* sigmoid_table) {
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
    } else if (l_type == POOLING) {
        PRINT_MSG("\nPooling.\n");
        result_loc = pooling_layer(activations, layers, layer_num, result);
    }

    PRINT_MSG("Result of layer %d:\n", layer_num);
    PRINT_DEBUG4D(result_loc, curr_layer.output_rows,
                  curr_layer.output_cols + curr_layer.output_data_align_pad,
                  curr_layer.output_height);

    return result_loc;
}
