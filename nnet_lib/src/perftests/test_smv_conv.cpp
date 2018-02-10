#include <math.h>
#include <stdio.h>
#include <string.h>

#include "config.h"
#include "core/nnet_fwd_defs.h"
#include "core/smv/smv.h"
#include "core/smv/params.h"
#include "utility/data_layout_conversion.h"
#include "utility/init_data.h"
#include "utility/utility.h"

int INPUT_DIM;
int NUM_CLASSES;
int NUM_TEST_CASES = 1;
float* sigmoid_table = NULL;
float* exp_table = NULL;
sigmoid_impl_t SIGMOID_IMPL;

int main(int argc, const char* argv[]) {
    layer_t curr_layer;
    curr_layer.type = CONV_STANDARD;
    curr_layer.inputs.height = 20;
    curr_layer.inputs.rows = 8;
    curr_layer.inputs.cols = 8;
    curr_layer.inputs.align_pad = 0;
    curr_layer.weights.height = curr_layer.inputs.height;
    curr_layer.weights.rows = 3;
    curr_layer.weights.cols = 3;
    curr_layer.weights.align_pad = 5;
    curr_layer.field_stride = 1;
    curr_layer.c_padding = 1;
    curr_layer.outputs.height = NUM_PE_INSTS;
    curr_layer.outputs.rows = curr_layer.inputs.rows;
    curr_layer.outputs.cols = curr_layer.inputs.cols;
    curr_layer.outputs.align_pad = curr_layer.inputs.align_pad;
    curr_layer.inputs.rows += 2 * curr_layer.c_padding;
    curr_layer.inputs.cols += 2 * curr_layer.c_padding;

    // device_t device;

    int total_input_size = get_input_activations_size(&curr_layer);
    int total_output_size = get_output_activations_size(&curr_layer);
    int total_weight_size = get_num_weights_layer(&curr_layer, 0);
    float* inputs = (float*)malloc_aligned(total_input_size * sizeof(float));
    float* weights = (float*)malloc_aligned(total_weight_size * sizeof(float));
    float* results = (float*)malloc_aligned(total_output_size * sizeof(float));
    // float* input_zeropad = NULL;
    assert(inputs && weights && results);

    // Initialize data.
    for (int i = 0; i < curr_layer.inputs.height; i++) {
        for (int j = 0; j < curr_layer.inputs.rows; j++) {
            for (int k = 0; k < curr_layer.inputs.cols; k++) {
                inputs[sub3ind(i, j, k, curr_layer.inputs.rows,
                               curr_layer.inputs.cols)] =
                        j * curr_layer.inputs.cols + k;
            }
        }
    }
    printf("NCHW inputs\n");
    print_debug4d(inputs, curr_layer.inputs.rows, curr_layer.inputs.cols,
                  curr_layer.inputs.height);
    for (int n = 0; n < curr_layer.outputs.height; n++) {
        for (int i = 0; i < curr_layer.weights.height; i++) {
            for (int j = 0; j < curr_layer.weights.rows; j++) {
                for (int k = 0; k < curr_layer.weights.cols; k++) {
                    weights[sub4ind(n, i, j, k, curr_layer.weights.height,
                                    curr_layer.weights.rows,
                                    curr_layer.weights.cols +
                                            curr_layer.weights.align_pad)] =
                            i + (j * curr_layer.weights.cols + k);
                }
            }
        }
    }
    printf("NCHW weights\n");
    print_debug4d(weights, curr_layer.weights.rows,
                  curr_layer.weights.cols + curr_layer.weights.align_pad,
                  curr_layer.weights.height);

    float* nhwc_activations = NULL;
    dims_t activations_nhwc = convert_nchw_to_nhwc(
            inputs, 1, curr_layer.inputs, DATA_ALIGNMENT, &nhwc_activations);

    printf("NHWC inputs\n");
    print_debug4d(nhwc_activations, activations_nhwc.rows,
                  activations_nhwc.cols + activations_nhwc.align_pad,
                  activations_nhwc.height);
    float* nhwc_weights = NULL;
    dims_t weights_nhwc = convert_nchw_to_nhwc(
            weights, NUM_PE_INSTS,
            curr_layer.weights, DATA_ALIGNMENT, &nhwc_weights);
    printf("NHWC weights\n");
    print_debug4d(nhwc_weights, weights_nhwc.rows,
                  weights_nhwc.cols + weights_nhwc.align_pad,
                  weights_nhwc.height);

    float* nhwc_results = NULL;
    dims_t nhwc_dims = nchw_to_nhwc_dims(&curr_layer.outputs, DATA_ALIGNMENT);
    nhwc_results = (float*)malloc_aligned(get_dims_size(&nhwc_dims) * sizeof(float));

    curr_layer.inputs.align_pad =
            calc_padding(curr_layer.inputs.height, DATA_ALIGNMENT);
    curr_layer.weights.align_pad =
            calc_padding(curr_layer.weights.height, DATA_ALIGNMENT);
    convolution3d_smv(nhwc_activations, nhwc_weights, curr_layer, 0, nhwc_results);

    printf("Results\n");
    print_debug4d(nhwc_results, curr_layer.outputs.rows,
                  curr_layer.outputs.cols + curr_layer.outputs.align_pad,
                  curr_layer.outputs.height);

    return 0;
}
