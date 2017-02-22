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

size_t calc_layer_intermediate_memory(layer_t layer) {
    size_t usage = 0;

    switch (layer.type) {
        case FC:
        case SOFTMAX:
            usage = layer.output_rows * layer.output_cols;
            break;
        case CONV:
        case POOL_MAX:
        case POOL_AVG:
            usage = max(
                    layer.input_rows * layer.input_cols * layer.input_height,
                    layer.output_rows * layer.output_cols *
                            layer.output_height);
            break;
        default:
            usage = 0;
            break;
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
           ARCHITECTURE == MONOLITHIC
                   ? "MONOLITHIC"
                   : ARCHITECTURE == COMPOSABLE ? "COMPOSABLE" : "UNKNOWN");
}

// This is the thing that we want to be good at in hardware
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

    layer_t* layers;
    int total_layers = configure_network_from_file(conf_file, &layers);
    printf("Size of layer configuration: %lu bytes\n",
           total_layers * sizeof(layer_t));

    data_init_mode RANDOM_DATA = RANDOM;
    data_init_mode RANDOM_WEIGHTS = RANDOM;

    // hid and hid_temp are the two primary buffers that will store the input
    // and output of each layer. They alternate in which one is input and which
    // is output. All input activations are initially loaded into hid. For this
    // reason, hid and hid_temp may not be the same size; hid must be large
    // enough to store the input activations, but this is not a concern for
    // hid_temp.
    float* hid;
    float* hid_temp;
    size_t data_size = NUM_TEST_CASES * INPUT_DIM;

    printf("Setting up arrays\n");
    // Get the dimensions of the biggest matrix that will ever come out of
    // run_layer.
    size_t hid_temp_size = 0;
    for (i = 0; i < total_layers; i++) {
        size_t curr_layer_usage = calc_layer_intermediate_memory(layers[i]);
        hid_temp_size = max(hid_temp_size, curr_layer_usage);
    }
    printf("  Largest intermediate output size is %lu elements\n",
           hid_temp_size);
    err = posix_memalign(
            (void**)&hid_temp, CACHELINE_SIZE,
            next_multiple(hid_temp_size * sizeof(float), CACHELINE_SIZE));
    ASSERT_MEMALIGN(hid_temp, err);
    size_t hid_size = max(data_size, hid_temp_size);
    printf("  hid has %lu elements\n", hid_size);
    err = posix_memalign(
            (void**)&hid, CACHELINE_SIZE,
            next_multiple(hid_size * sizeof(float), CACHELINE_SIZE));
    ASSERT_MEMALIGN(hid, err);

    // Initialize weights, data, and labels.
    float* weights;
    int w_size = get_total_num_weights(layers, total_layers);
    err = posix_memalign((void**)&weights, CACHELINE_SIZE,
                         next_multiple(w_size * sizeof(float), CACHELINE_SIZE));
    ASSERT_MEMALIGN(weights, err);
    printf("  Total weights: %d elements\n", w_size);
    // Get the largest weights size for a single layer - this will be the size
    // of the scratchpad.
    size_t weights_temp_size = 0;
    for (i = 0; i < total_layers; i++) {
      size_t curr_layer_weights = get_num_weights_layer(layers, i);
      weights_temp_size = max(weights_temp_size, curr_layer_weights);
    }
    printf("  Largest weights per layer: %lu elements\n", weights_temp_size);

    init_weights(weights, layers, total_layers, RANDOM_WEIGHTS, TRANSPOSE_WEIGHTS);

    int* labels;
    size_t label_size = NUM_TEST_CASES;
    err = posix_memalign(
            (void**)&labels, CACHELINE_SIZE,
            next_multiple(label_size * sizeof(int), CACHELINE_SIZE));
    ASSERT_MEMALIGN(labels, err);

    init_data(hid, NUM_TEST_CASES, INPUT_DIM, RANDOM_DATA);
    init_labels(labels, NUM_TEST_CASES, RANDOM_DATA);

    // This file is not looked at by aladdin so malloc is fine.
    // If I do the old version then I get a memory overflow, because the
    // max stack size is not big enough for TIMIT stuff.

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

    // -------------------------------------------------------- //
    //     THIS IS THE FUNCTION BEING SIMULATED IN HARDWARE     //
    // -------------------------------------------------------- //
#ifdef GEM5_HARNESS
    mapArrayToAccelerator(
            INTEGRATION_TEST, "hid", hid, hid_size * sizeof(float));
    mapArrayToAccelerator(
            INTEGRATION_TEST, "hid_temp", hid_temp, hid_size * sizeof(float));
    mapArrayToAccelerator(
            INTEGRATION_TEST, "weights", weights, w_size * sizeof(float));
    mapArrayToAccelerator(
            INTEGRATION_TEST, "layers", layers, total_layers * sizeof(layer_t));
    invokeAcceleratorAndBlock(INTEGRATION_TEST);
#else
    // Run a forward pass through the neural net
    printf("Running forward pass\n");
    // The function being synthesized
    nnet_fwd(hid, weights, layers, total_layers, hid_temp, sigmoid_table);
#endif

    // Print the result, maybe not all the test_cases
    int num_to_print = 1;
    // don't try to print more test cases than there are
    num_to_print =
            num_to_print < NUM_TEST_CASES ? num_to_print : NUM_TEST_CASES;

    // Compute the classification error rate
    float* result = layers[total_layers-1].result_in_temp ? hid_temp : hid;
    int num_errors = 0;
    for (i = 0; i < NUM_TEST_CASES; i++) {
        if (arg_max(result + i * NUM_CLASSES, NUM_CLASSES, 1) != labels[i]) {
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

    free(hid);
    free(hid_temp);
    free(weights);
    free(labels);
    free(layers);
}
