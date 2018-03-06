#include "nnet_fwd.h"
#include "arch/common.h"
#include "arch/interface.h"
#include "utility/utility.h"

// Common dispatch function for executing a layer.
//
// Whether an activation function will be applied on the output or not depends
// on the backend.
//
// A result_buf value is returned indicating the final location of the output of
// this layer. It is either equal to the pointers @activations or @result.
result_buf layer_dispatcher(data_list* activations,
                            data_list* weights,
                            layer_t* layers,
                            int layer_num,
                            data_list* result,
                            device_t* device,
                            sampling_param_t* sampling_param) {
    layer_t curr_layer = layers[layer_num];
    layer_type l_type = curr_layer.type;
    result_buf result_loc = result;

    if (l_type == FC) {
        PRINT_MSG("\nInner product.\n");
        if (curr_layer.input_preprocessing == FLATTEN) {
            PRINT_MSG("Flattening the input.\n");
            result_loc = flatten_input(activations, layers, layer_num, result);
        }
        if (curr_layer.input_preprocessing == FLATTEN && result_loc == result) {
            PRINT_MSG("After flattening:\n");
            PRINT_DEBUG(result_loc->data[0].dense->d, NUM_TEST_CASES,
                        layers[layer_num].inputs.cols,
                        layers[layer_num].inputs.cols);
            result_loc = inner_product_layer(result,
                                             weights,
                                             layers,
                                             layer_num,
                                             activations,
                                             device,
                                             sampling_param);
        } else {
            result_loc = inner_product_layer(activations,
                                             weights,
                                             layers,
                                             layer_num,
                                             result,
                                             device,
                                             sampling_param);
        }
    } else if (l_type == CONV_STANDARD) {
        PRINT_MSG("\nStandard convolution.\n");
        result_loc = standard_convolution_layer(activations,
                                                weights,
                                                layers,
                                                layer_num,
                                                result,
                                                device,
                                                sampling_param);
    } else if (l_type == CONV_DEPTHWISE) {
        PRINT_MSG("\nDepthwise convolution.\n");
        result_loc = depthwise_convolution_layer(activations,
                                                 weights,
                                                 layers,
                                                 layer_num,
                                                 result,
                                                 device,
                                                 sampling_param);
    } else if (l_type == CONV_POINTWISE) {
        PRINT_MSG("\nPointwise convolution.\n");
        result_loc = pointwise_convolution_layer(activations,
                                                 weights,
                                                 layers,
                                                 layer_num,
                                                 result,
                                                 device,
                                                 sampling_param);
    } else if (l_type == POOLING) {
        PRINT_MSG("\nPooling.\n");
        result_loc = pooling_layer(
                activations, layers, layer_num, result, device, sampling_param);
    } else if (l_type == BATCH_NORM) {
        PRINT_MSG("\nBatch normalization.\n");
        result_loc = batch_norm_layer(activations,
                                      weights,
                                      layers,
                                      layer_num,
                                      result,
                                      device,
                                      sampling_param);
    } else if (l_type == INPUT) {
        // No work needs to be done.
        result_loc = activations;
        return result_loc;
    }

    PRINT_MSG("Result of layer %d:\n", layer_num);
    if (result_loc->type[0] == Uncompressed) {
        PRINT_DEBUG4D(result_loc->data[0].dense->d, curr_layer.outputs.rows,
                      curr_layer.outputs.cols + curr_layer.outputs.align_pad,
                      curr_layer.outputs.height);
    } else if (result_loc->type[0] == UncompressedHalfPrecision) {
        PRINT_DEBUG4D_FP16(
                result_loc->data[0].dense_hp->d, NUM_TEST_CASES,
                curr_layer.outputs.height, curr_layer.outputs.rows,
                curr_layer.outputs.cols + curr_layer.outputs.align_pad);
    } else {
        PRINT_MSG("Unable to print results in compressed form.\n");
    }

    return result_loc;
}
