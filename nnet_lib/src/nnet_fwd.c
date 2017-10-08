#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

#include "core/activation_functions.h"
#include "core/convolution.h"
#include "core/matrix_multiply.h"
#include "core/pooling.h"
#include "core/zeropad.h"
#include "utility/init_data.h"
#include "utility/read_model_conf.h"
#include "utility/utility.h"
#include "arch/interface.h"
#include "arch/common.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

#ifdef GEM5_HARNESS
#include "gem5/aladdin_sys_connection.h"
#include "gem5/aladdin_sys_constants.h"
#endif

#include "nnet_fwd.h"

int NUM_TEST_CASES;
int NUM_CLASSES;
int INPUT_DIM;

void get_input_data(float* input) {
#ifdef DMA_MODE
    dmaLoad(input, 0, 0, NUM_TEST_CASES * INPUT_DIM * sizeof(float));
#endif
}

void store_result(float* result,
                  float* result_temp,
                  layer_t* layers,
                  int num_layers,
                  bool result_in_temp) {
#ifdef DMA_MODE
    if (result_in_temp)
        dmaStore(result_temp, 0, 0, NUM_TEST_CASES * NUM_CLASSES * sizeof(float));
    else
        dmaStore(result, 0, 0, NUM_TEST_CASES * NUM_CLASSES * sizeof(float));
    dmaStore(layers, 0, 0, num_layers*sizeof(layer_t));
#endif
}

size_t calc_layer_intermediate_memory(layer_t* layers, int lnum) {
    size_t usage = 0, flattened_usage = 0;
    layer_t layer = layers[lnum];

    if (layer.type == INPUT) {
        usage = layer.inputs.rows *
                (layer.inputs.cols + layer.inputs.align_pad) *
                layer.inputs.height;
    } else if (layer.type == FC || layer.type == SOFTMAX) {
        usage = layer.outputs.rows *
                (layer.outputs.cols + layer.outputs.align_pad);
        if (layer.input_preprocessing == FLATTEN) {
            // Flattening the input will require the second buffer.
            layer_t prev_layer = layers[lnum];
            flattened_usage =
                    prev_layer.outputs.rows *
                    (prev_layer.outputs.cols + prev_layer.outputs.align_pad) *
                    prev_layer.outputs.height;
            usage = max(usage, flattened_usage);
        } else {
            usage = layer.outputs.rows *
                    (layer.outputs.cols + layer.outputs.align_pad);
        }
    } else if (layer.type == CONV || layer.type == POOLING) {
        usage = max(layer.inputs.rows *
                            (layer.inputs.cols + layer.inputs.align_pad) *
                            layer.inputs.height,
                    layer.outputs.rows *
                            (layer.outputs.cols + layer.outputs.align_pad) *
                            layer.outputs.height);
    } else {
        usage = 0;
    }
    return usage * NUM_TEST_CASES;
}

void print_usage() {
    printf("Usage:\n");
    printf("  nnet_fwd path/to/model-config-file [num-inputs=1]\n\n");
    printf("  The model configuration file is written in libconfuse syntax,\n "
           "    based loosely on the Caffe configuration style. It is case\n"
           "    sensitive.\n\n");
    printf("  num-inputs specifies the number of input images to run through\n"
           "    the network. If not specified, it defaults to 1.\n\n");
    printf("Build type: %s\n",
           ARCHITECTURE == MONOLITHIC ? "MONOLITHIC" :
           ARCHITECTURE == COMPOSABLE ? "COMPOSABLE" :
           ARCHITECTURE == SMIV ? "SMIV" :
           "UNKNOWN");
}

int main(int argc, const char* argv[]) {
    int i, j, err;

    if (argc < 2 || argc > 3) {
      print_usage();
      return -1;
    }
    const char* conf_file = argv[1];
    if (argc == 2)
      NUM_TEST_CASES = 1;
    else
      NUM_TEST_CASES = strtol(argv[2], NULL, 10);

    // set random seed (need to #include <time.h>)
    srand(1);

    network_t network;
    network.depth = configure_network_from_file(conf_file, &network.layers);
    printf("Size of layer configuration: %lu bytes\n",
           network.depth * sizeof(layer_t));

    data_init_mode RANDOM_DATA = RANDOM;
    data_init_mode RANDOM_WEIGHTS = RANDOM;

    // hid and hid_temp are the two primary buffers that will store the input
    // and output of each layer. They alternate in which one is input and which
    // is output. All input activations are initially loaded into hid. For this
    // reason, hid and hid_temp may not be the same size; hid must be large
    // enough to store the input activations, but this is not a concern for
    // hid_temp.
    farray_t hid = { NULL, 0 };
    farray_t hid_temp = { NULL, 0 };
    layer_t input_layer = network.layers[0];
    size_t data_size =
            NUM_TEST_CASES * input_layer.inputs.rows *
            (input_layer.inputs.cols + input_layer.inputs.align_pad) *
            input_layer.inputs.height;

    printf("Setting up arrays\n");
    // Get the dimensions of the biggest matrix that will ever come out of
    // run_layer.
    for (i = 0; i < network.depth; i++) {
        size_t curr_layer_usage =
                calc_layer_intermediate_memory(network.layers, i);
        hid_temp.size = max(hid_temp.size, curr_layer_usage);
    }
    printf("  Largest intermediate output size is %lu elements\n",
           hid_temp.size);
    err = posix_memalign(
            (void**)&hid_temp.d, CACHELINE_SIZE,
            next_multiple(hid_temp.size * sizeof(float), CACHELINE_SIZE));
    ASSERT_MEMALIGN(hid_temp.d, err);
    hid.size = max(data_size, hid_temp.size);
    printf("  hid has %lu elements\n", hid.size);
    err = posix_memalign(
            (void**)&hid.d, CACHELINE_SIZE,
            next_multiple(hid.size * sizeof(float), CACHELINE_SIZE));
    ASSERT_MEMALIGN(hid.d, err);

    // Initialize weights, data, and labels.
    farray_t weights;
    weights.size = get_total_num_weights(network.layers, network.depth);
    err = posix_memalign(
            (void**)&weights.d, CACHELINE_SIZE,
            next_multiple(weights.size * sizeof(float), CACHELINE_SIZE));
    ASSERT_MEMALIGN(weights.d, err);
    printf("  Total weights: %lu elements\n", weights.size);
    // Get the largest weights size for a single layer - this will be the size
    // of the scratchpad.
    size_t weights_temp_size = 0;
    for (i = 0; i < network.depth; i++) {
        size_t curr_layer_weights = get_num_weights_layer(network.layers, i);
        weights_temp_size = max(weights_temp_size, curr_layer_weights);
    }
    printf("  Largest weights per layer: %lu elements\n", weights_temp_size);

    init_weights(weights.d, network.layers, network.depth, RANDOM_WEIGHTS,
                 TRANSPOSE_WEIGHTS);

    iarray_t labels = { NULL, 0 };
    labels.size = NUM_TEST_CASES;
    err = posix_memalign(
            (void**)&labels.d, CACHELINE_SIZE,
            next_multiple(labels.size * sizeof(int), CACHELINE_SIZE));
    ASSERT_MEMALIGN(labels.d, err);

    init_data(hid.d, &network, NUM_TEST_CASES, RANDOM_DATA);
    init_labels(labels.d, NUM_TEST_CASES, RANDOM_DATA);

    // Build the sigmoid lookup table
    // May want to change this to be "non-centered"
    // to avoid (sigmoid_coarseness - 1.0)
    // so we can use bit shift in lookup function with fixed point precisions
    printf("Setting up sigmoid lookup table\n");
    int sigmoid_coarseness = 1 << LG_SIGMOID_COARSENESS;
    float sigmoid_table[sigmoid_coarseness];
    float sig_step = (float)(SIG_MAX - SIG_MIN) / (sigmoid_coarseness - 1.0);
    float x_sig = (float)SIG_MIN;
    for (i = 0; i < sigmoid_coarseness; i++) {
        sigmoid_table[i] = conv_float2fixed(1.0 / (1.0 + exp(-x_sig)));
        // printf("%f, %f\n", x_sig, sigmoid_table[i]);
        x_sig += sig_step;
    }

    fflush(stdout);

    // Run a forward pass through the neural net
    printf("Running forward pass\n");
    nnet_fwd(hid, weights, hid_temp, network, sigmoid_table);

    // Print the result, maybe not all the test_cases
    int num_to_print = 1;
    // don't try to print more test cases than there are
    num_to_print =
            num_to_print < NUM_TEST_CASES ? num_to_print : NUM_TEST_CASES;

    // Compute the classification error rate
    float* result = network.layers[network.depth - 1].result_in_temp
                            ? hid_temp.d
                            : hid.d;
    int num_errors = 0;
    for (i = 0; i < NUM_TEST_CASES; i++) {
        if (arg_max(result + i * NUM_CLASSES, NUM_CLASSES, 1) != labels.d[i]) {
            num_errors = num_errors + 1;
        }
    }
    float error_fraction = ((float)num_errors) / ((float)NUM_TEST_CASES);
    printf("Fraction incorrect (over %d cases) = %f\n", NUM_TEST_CASES,
           error_fraction);

    // Print the output labels and soft outputs.
    FILE* output_labels = fopen("output_labels.out", "w");
    for (i = 0; i < NUM_TEST_CASES; i++) {
        int pred = arg_max(result + i * NUM_CLASSES, NUM_CLASSES, 1);
        fprintf(output_labels, "Test %d: %d\n  [", i, pred);
        for (j = 0; j < NUM_CLASSES; j++)
            fprintf(output_labels, "%f  ", result[sub2ind(i, j, NUM_CLASSES)]);
        fprintf(output_labels, "]\n");
    }
    fclose(output_labels);

    free(hid.d);
    free(hid_temp.d);
    free(weights.d);
    free(labels.d);
    free(network.layers);
}
