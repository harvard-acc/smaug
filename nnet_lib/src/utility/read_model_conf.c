#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "confuse.h"

#include "utility/utility.h"
#include "nnet_fwd.h"

#include "read_model_conf.h"

extern cfg_opt_t convolution_param_cfg[];
extern cfg_opt_t inner_product_param_cfg[];
extern cfg_opt_t pooling_param_cfg[];
extern cfg_opt_t layer_cfg[];
extern cfg_opt_t network_cfg[];
extern cfg_opt_t top_level_cfg[];

const char CONV_TYPE[] = "CONVOLUTION";
const char FC_TYPE[] = "INNER_PRODUCT";
const char POOLING_TYPE[] = "POOLING";
const char MAX_POOL_TYPE[] = "MAX";
const char AVG_POOL_TYPE[] = "AVG";
const char NONE_TYPE[] = "NONE";
const char RELU_TYPE[] = "RELU";
const char SIGMOID_TYPE[] = "SIGMOID";

static int input_rows;
static int input_cols;
static int input_height;
static int data_alignment;

static void set_layer_type(layer_t* layers, cfg_t* layer_opts, int l) {
    const char* type = cfg_getstr(layer_opts, "type");
    if (strcmp(type, CONV_TYPE) == 0) {
        layers[l].type = CONV;
    } else if (strcmp(type, POOLING_TYPE) == 0) {
        layers[l].type = POOLING;
        cfg_t* pool_cfg = cfg_getsec(layer_opts, "pooling_param");
        const char* pool_type_str = cfg_getstr(pool_cfg, "pool");
        if (strcmp(pool_type_str, MAX_POOL_TYPE) == 0) {
            layers[l].pool = MAX;
        } else if (strcmp(pool_type_str, AVG_POOL_TYPE) == 0) {
            layers[l].pool = AVG;
        } else {
            assert(false && "Invalid type of pooling layer!");
        }
    } else if (strcmp(type, FC_TYPE) == 0) {
        layers[l].type = FC;
    } else {
        assert(false && "Invalid layer type!");
    }

    const char* activation = cfg_getstr(layer_opts, "activation");
    if (strcmp(activation, RELU_TYPE) == 0) {
        layers[l].activation = RELU;
    } else if (strcmp(activation, SIGMOID_TYPE) == 0) {
        layers[l].activation = SIGMOID;
    } else if (strcmp(activation, NONE_TYPE) == 0) {
        layers[l].activation = NO_ACTIVATION;
    } else {
        assert(false && "Invalid activation type!");
    }

    if (layers[l].type == POOLING && layers[l].activation != NO_ACTIVATION)
        fprintf(stderr, "Pooling layer %d has an activation function, which is "
                        "usually unnecessary.\n",
                l);
}

static void set_layer_dims(layer_t* layers, cfg_t* layer_opts, int l) {
    if (layers[l].type == CONV) {
        layers[l].inputs.rows = layers[l - 1].outputs.rows;
        layers[l].inputs.cols = layers[l - 1].outputs.cols;
        layers[l].inputs.height = layers[l - 1].outputs.height;

        cfg_t* conv_params = cfg_getsec(layer_opts, "convolution_param");
        layers[l].weights.rows = cfg_getint(conv_params, "kernel_size");
        layers[l].weights.cols = cfg_getint(conv_params, "kernel_size");
        layers[l].weights.height = layers[l].inputs.height;
        layers[l].field_stride = cfg_getint(conv_params, "stride");

        layers[l].c_padding = cfg_getint(conv_params, "pad");
        layers[l].inputs.rows += layers[l].c_padding * 2;
        layers[l].inputs.cols += layers[l].c_padding * 2;

        layers[l].outputs.rows =
                (layers[l].inputs.rows - layers[l].weights.cols) /
                        layers[l].field_stride + 1;
        layers[l].outputs.cols =
                (layers[l].inputs.cols - layers[l].weights.cols) /
                        layers[l].field_stride + 1;
        // Number of kernels is the third dimension of the output.
        layers[l].outputs.height = cfg_getint(conv_params, "num_output");

        assert(layers[l].weights.rows != -1);
        assert(layers[l].c_padding != -1);
        assert(layers[l].outputs.height != -1);
        return;
    }

    if (layers[l].type == FC) {
        cfg_t* fc_params = cfg_getsec(layer_opts, "inner_product_param");
        layers[l].inputs.rows = 1;
        if (layers[l - 1].type != FC) {
            // If the previous layer was not an FC layer, we have to flatten
            // the previous layer's output.
            layers[l].inputs.cols = layers[l-1].outputs.rows *
                                    layers[l-1].outputs.cols *
                                    layers[l-1].outputs.height;
        } else {
            layers[l].inputs.cols = layers[l-1].outputs.cols;
        }

        layers[l].weights.rows = layers[l].inputs.cols + 1;  // for bias.
        layers[l].weights.cols = cfg_getint(fc_params, "num_output");

        layers[l].outputs.rows = layers[l].inputs.rows;
        layers[l].outputs.cols = layers[l].weights.cols;

        layers[l].inputs.height = 1;
        layers[l].outputs.height = 1;
        layers[l].weights.height = 1;
        return;
    }

    if (layers[l].type == POOLING) {
      layers[l].inputs.rows = layers[l - 1].outputs.rows;
      layers[l].inputs.cols = layers[l - 1].outputs.cols;
      layers[l].inputs.height = layers[l - 1].outputs.height;

      cfg_t* pool_params = cfg_getsec(layer_opts, "pooling_param");
      layers[l].weights.rows = cfg_getint(pool_params, "size");
      layers[l].weights.cols = cfg_getint(pool_params, "size");
      layers[l].weights.height = 0;  // Not used.
      layers[l].field_stride = cfg_getint(pool_params, "stride");

      layers[l].outputs.rows = (layers[l].inputs.rows - layers[l].weights.cols) /
                                       layers[l].field_stride +
                               1;
      layers[l].outputs.cols = (layers[l].inputs.cols - layers[l].weights.cols) /
                                       layers[l].field_stride +
                               1;
      layers[l].outputs.height = layers[l].inputs.height;
      assert(layers[l].weights.rows != -1);
      assert(layers[l].field_stride != -1);
      return;
    }

    if (layers[l].type == SOFTMAX) {
      layers[l].inputs.rows = layers[l - 1].outputs.rows;
      layers[l].inputs.cols = layers[l - 1].outputs.cols;
      layers[l].inputs.height = layers[l - 1].outputs.height;
      layers[l].weights.rows = 0;
      layers[l].weights.cols = 0;
      layers[l].weights.height = 0;
      layers[l].outputs.rows = layers[l].inputs.rows;
      layers[l].outputs.cols = layers[l].inputs.cols;
      layers[l].outputs.height = layers[l].inputs.height;
      return;
    }
}

static void handle_data_alignment(layer_t* layers, int l) {
    if (layers[l].input_preprocessing == UNFLATTEN) {
        // When unflattening, we need to align each row, not the flattened
        // dimension, of the input.
        layers[l].inputs.align_pad =
                calc_padding(layers[l - 1].outputs.cols, data_alignment);
        // The output is aligned as usual.
        layers[l].outputs.align_pad =
                calc_padding(layers[l].outputs.cols, data_alignment);
    } else {
        // The input pad amount is determined by the number of
        // desired cols for this layer.
        layers[l].inputs.align_pad =
                calc_padding(layers[l].inputs.cols, data_alignment);
        layers[l].outputs.align_pad =
                calc_padding(layers[l].outputs.cols, data_alignment);
    }
    layers[l].weights.align_pad =
            calc_padding(layers[l].weights.cols, data_alignment);
}

static void read_top_level_config(layer_t* layers, cfg_t* network_opts) {
    layers[0].inputs.rows = cfg_getint(network_opts, "input_rows");
    layers[0].inputs.cols = cfg_getint(network_opts, "input_cols");
    layers[0].inputs.height = cfg_getint(network_opts, "input_height");
    layers[0].type = INPUT;
    layers[0].activation = NO_ACTIVATION;
    layers[0].outputs.rows = layers[0].inputs.rows;
    layers[0].outputs.cols = layers[0].inputs.cols;
    layers[0].outputs.height = layers[0].inputs.height;
    layers[0].weights.rows = 0;
    layers[0].weights.cols = 0;
    layers[0].weights.height = 0;

    // Set the global variables.
    data_alignment = DATA_ALIGNMENT;
    input_rows = layers[0].inputs.rows;
    input_cols = layers[0].inputs.cols;
    input_height = layers[0].inputs.height;
}

static void read_layer_config(layer_t* layers, cfg_t* network_opts, int l) {
    cfg_t* current_layer_opts = cfg_getnsec(network_opts, "layer", l - 1);
    set_layer_type(layers, current_layer_opts, l);
    set_layer_dims(layers, current_layer_opts, l);
}

static void print_layer_config(layer_t* layers, int num_layers) {
    printf("==================================\n");
    printf("Network configuration (per input):\n");
    printf("----------------------------------\n");
    for (int i = 0; i < num_layers; i++) {
        layer_type type = layers[i].type;
        activation_type act = layers[i].activation;
        printf("  Layer %d: ", i);
        if (type == CONV) {
            printf("  Convolutional\n");
            printf("    Input size: %d x %d x %d (after padding)\n",
                   layers[i].inputs.rows, layers[i].inputs.cols,
                   layers[i].inputs.height);
            printf("    Output size: %d x %d x %d\n", layers[i].outputs.rows,
                   layers[i].outputs.cols, layers[i].outputs.height);
            printf("    Kernel size: %d x %d x %d\n", layers[i].weights.cols,
                   layers[i].weights.cols, layers[i].inputs.height);
            printf("    Num kernels: %d\n", layers[i].outputs.height);
            printf("    Padding: %d\n", layers[i].c_padding);
            printf("    Stride: %d\n", layers[i].field_stride);
        } else if (type == FC) {
            printf("  Fully connected\n");
            printf("    Input size: %d x %d\n", layers[i].inputs.rows,
                   layers[i].inputs.cols);
            printf("    Weights: %d x %d\n", layers[i].weights.rows,
                   layers[i].weights.cols);
        } else if (type == POOLING) {
            printf("  Max pooling\n");
            printf("    Input size: %d x %d x %d\n", layers[i].inputs.rows,
                   layers[i].inputs.cols, layers[i].inputs.height);
            printf("    Output size: %d x %d x %d\n", layers[i].outputs.rows,
                   layers[i].outputs.cols, layers[i].outputs.height);
            printf("    Field size: %d\n", layers[i].weights.cols);
            printf("    Stride: %d\n", layers[i].field_stride);
            printf("    Height: %d\n", layers[i].outputs.height);
        } else if (type == SOFTMAX) {
            printf("  Softmax\n");
        } else if (type == INPUT) {
            printf("  Input layer\n");
            printf("    Input size: %d x %d x %d\n", layers[i].inputs.rows,
                   layers[i].inputs.cols, layers[i].inputs.height);
        }
        printf("    Input data padding: %d\n", layers[i].inputs.align_pad);
        printf("    Output data padding: %d\n", layers[i].outputs.align_pad);
        printf("    Activation: %s\n",
               act == RELU ? "RELU" : act == SIGMOID ? "SIGMOID" : "NONE");
    }
    printf("==================================\n");
}

int configure_network_from_file(const char* cfg_file, layer_t** layers_ptr) {
    cfg_t* all_opts = cfg_init(top_level_cfg, CFGF_NONE);
    int ret = cfg_parse(all_opts, cfg_file);
    if (ret == CFG_FILE_ERROR) {
        assert(false && "Failed to open configuration file!");
    } else if (ret == CFG_PARSE_ERROR) {
        assert(false && "Config parsing error");
    }

    cfg_t* network_opts = cfg_getsec(all_opts, "network");
    int num_layers = cfg_size(network_opts, "layer") + 1;  // +1 for input layer.

    int err = posix_memalign(
            (void**)layers_ptr, CACHELINE_SIZE,
            next_multiple(sizeof(layer_t) * num_layers, CACHELINE_SIZE));
    ASSERT_MEMALIGN(*layers_ptr, err);

    layer_t* layers = *layers_ptr;

    //=---------------------  STEP 1 -----------------------=//
    // First, read in all the parameters from the configuration
    // file for each layer.

    read_top_level_config(layers, network_opts);
    for (int i = 1; i < num_layers; i++) {
        read_layer_config(layers, network_opts, i);
    }

    //=---------------------  STEP 2 -----------------------=//
    // Identify layers that require their input to be flattened
    // (CONV/INPUT to FC) or unflattened (FC to CONV).

    layers[0].input_preprocessing = NO_PREPROCESSING;
    for (int i = 1; i < num_layers; i++) {
        if (layers[i].type == FC && layers[i-1].type != FC) {
            layers[i].input_preprocessing = FLATTEN;
        } else if (layers[i].type == CONV && layers[i-1].type == FC) {
            layers[i].input_preprocessing = UNFLATTEN;
        } else {
            layers[i].input_preprocessing = NO_PREPROCESSING;
        }
    }

    //=---------------------  STEP 3 -----------------------=//
    // Compute data alignment requirements for each layer's
    // inputs and weights. This needs to account for flattening.

    for (int i = 0; i < num_layers; i++) {
        handle_data_alignment(layers, i);
    }

    // Set some global variables.
    NUM_CLASSES = layers[num_layers-1].outputs.cols;
    INPUT_DIM = input_rows * input_cols * input_height;

    assert(INPUT_DIM > 0);
    assert(NUM_CLASSES > 0);

    print_layer_config(*layers_ptr, num_layers);
    cfg_free(all_opts);
    return num_layers;
}
