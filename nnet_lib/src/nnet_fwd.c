#include <argp.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "core/activation_functions.h"
#include "core/convolution.h"
#include "core/matrix_multiply.h"
#include "core/pooling.h"
#include "core/zeropad.h"
#include "utility/init_data.h"
#include "utility/read_model_conf.h"
#include "utility/utility.h"
#include "utility/profiling.h"
#include "utility/data_archive.h"
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
float* sigmoid_table;

static char prog_doc[] =
        "\nNeural network library for gem5-aladdin.\n"
        "   The model configuration file is written in libconfuse syntax,\n"
        "   based loosely on the Caffe configuration style. It is case\n"
        "   sensitive.\n\n"
        "Build type: " ARCH_STR "\n";
static char args_doc[] = "path/to/model-config-file";
static struct argp_option options[] = {
    { "num-inputs", 'n', "N", 0, "Number of input images" },
    { "data-init-mode", 'd', "D", 0,
      "Data and weights generation mode (FIXED, RANDOM, READ_FILE)." },
    { "data-file", 'f', "F", 0,
      "File to read data and weights from (if data-init-mode == READ_FILE or "
      "save-params is true)." },
    { "save-params", 's', 0, 0,
      "Save network weights, data, and labels to a file." },
    { 0 },
};

typedef enum _argnum {
    NETWORK_CONFIG,
    NUM_REQUIRED_ARGS,
    DATA_FILE = NUM_REQUIRED_ARGS,
    NUM_ARGS,
} argnum;

typedef struct _arguments {
    char* args[NUM_ARGS];
    int num_inputs;
    bool save_params;
    data_init_mode data_mode;
} arguments;

int str2mode(char* str, data_init_mode* mode) {
    if (strncmp(str, "RANDOM", 6) == 0) {
        *mode = RANDOM;
        return 0;
    } else if (strncmp(str, "FIXED", 5) == 0) {
        *mode = FIXED;
        return 0;
    } else if (strncmp(str, "READ_FILE", 9) == 0) {
        *mode = READ_FILE;
        return 0;
    }
    return 1;
}

static error_t parse_opt(int key, char* arg, struct argp_state* state) {
  arguments *args = (arguments*)(state->input);
  switch (key) {
    case 'n': {
      args->num_inputs = strtol(arg, NULL, 10);
      break;
    }
    case 'd': {
      if (str2mode(arg, &args->data_mode))
        argp_usage(state);
      break;
    }
    case 'f': {
      args->args[DATA_FILE] = arg;
      break;
    }
    case 's': {
      args->save_params = true;
      break;
    }
    case ARGP_KEY_ARG: {
      if (state->arg_num >= NUM_REQUIRED_ARGS)
        argp_usage(state);
      args->args[state->arg_num] = arg;
      break;
    }
    case ARGP_KEY_END: {
        if (state->arg_num < NUM_REQUIRED_ARGS)
            argp_usage(state);
        break;
    }
    case ARGP_KEY_FINI: {
        if (args->data_mode == READ_FILE && !args->args[DATA_FILE]) {
            fprintf(stderr,
                    "[ERROR]: You must specify a data file to read parameters "
                    "from.\n");
            argp_usage(state);
        }
        if (args->save_params && !args->args[DATA_FILE]) {
            fprintf(stderr,
                    "[ERROR]: You must specify a data file to save parameters "
                    "to.\n");
            argp_usage(state);
        }
        break;
    }
    default:
      return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static struct argp parser = {options, parse_opt, args_doc, prog_doc};

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

void set_default_args(arguments* args) {
    args->num_inputs = 1;
    args->data_mode = RANDOM;
    args->save_params = false;
    for (int i = 0; i < NUM_ARGS; i++) {
        args->args[i] = NULL;
    }
}

int main(int argc, char* argv[]) {
    int i, j;

    arguments args;
    set_default_args(&args);
    argp_parse(&parser, argc, argv, 0, 0, &args);

    NUM_TEST_CASES = args.num_inputs;
    printf("Batch size: %d\n", NUM_TEST_CASES);

    // set random seed (need to #include <time.h>)
    srand(1);

    network_t network;
    device_t* device;
    network.depth =
            configure_network_from_file(args.args[0], &network.layers, &device);
    printf("Size of layer configuration: %lu bytes\n",
           network.depth * sizeof(layer_t));

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
    hid_temp.d = (float*)malloc_aligned(hid_temp.size * sizeof(float));
    memset(hid_temp.d, 0, hid_temp.size * sizeof(float));

    hid.size = max(data_size, hid_temp.size);
    printf("  hid has %lu elements\n", hid.size);
    hid.d = (float*)malloc_aligned(hid.size * sizeof(float));
    memset(hid.d, 0, hid.size * sizeof(float));

    // Initialize weights, data, and labels.
    farray_t weights;
    weights.size = get_total_num_weights(network.layers, network.depth);
    weights.d = (float*)malloc_aligned(weights.size * sizeof(float));
    memset(weights.d, 0, weights.size * sizeof(float));
    printf("  Total weights: %lu elements\n", weights.size);
    // Get the largest weights size for a single layer - this will be the size
    // of the scratchpad.
    size_t weights_temp_size = 0;
    for (i = 0; i < network.depth; i++) {
        size_t curr_layer_weights = get_num_weights_layer(network.layers, i);
        weights_temp_size = max(weights_temp_size, curr_layer_weights);
    }
    printf("  Largest weights per layer: %lu elements\n", weights_temp_size);

    iarray_t labels = { NULL, 0 };
    labels.size = NUM_TEST_CASES;
    labels.d = (int*)malloc_aligned(labels.size * sizeof(float));
    memset(labels.d, 0, labels.size * sizeof(float));

    if (args.data_mode == READ_FILE) {
        verify_global_parameters(args.args[DATA_FILE], &network);
        read_weights_from_file(args.args[DATA_FILE], &weights);
        read_data_from_file(args.args[DATA_FILE], &hid);
        read_labels_from_file(args.args[DATA_FILE], &labels);
    } else {
        init_weights(weights.d, network.layers, network.depth, args.data_mode,
                     TRANSPOSE_WEIGHTS);
        init_data(hid.d, &network, NUM_TEST_CASES, args.data_mode);
        init_labels(labels.d, NUM_TEST_CASES, args.data_mode);
    }

    if (args.save_params) {
        FILE* network_dump = fopen(args.args[DATA_FILE], "w");
        save_global_parameters(network_dump, &network);
        save_weights(network_dump, weights, weights.size);
        save_data(network_dump, hid, INPUT_DIM * NUM_TEST_CASES);
        save_labels(network_dump, labels, labels.size);
        fclose(network_dump);
    }

    // Build the sigmoid lookup table
    // May want to change this to be "non-centered"
    // to avoid (sigmoid_coarseness - 1.0)
    // so we can use bit shift in lookup function with fixed point precisions
    printf("Setting up sigmoid lookup table\n");
    int sigmoid_coarseness = 1 << LG_SIGMOID_COARSENESS;
    sigmoid_table = (float*)malloc(sigmoid_coarseness * sizeof(float));
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
    init_profiling_log();
    nnet_fwd(hid, weights, hid_temp, network, device);
    dump_profiling_log();
    close_profiling_log();

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

    free(sigmoid_table);
    free(hid.d);
    free(hid_temp.d);
    free(weights.d);
    free(labels.d);
    free(network.layers);
    free(device);
}
