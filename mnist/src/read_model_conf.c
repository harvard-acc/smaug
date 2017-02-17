#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "confuse.h"
#include "nnet_fwd.h"
#include "read_model_conf.h"
#include "utility.h"

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

static int input_rows;
static int input_cols;
static int input_height;

static void set_layer_type(layer_t* layers, cfg_t* layer_opts, int l) {
    const char* type = cfg_getstr(layer_opts, "type");
    if (strcmp(type, CONV_TYPE) == 0) {
        layers[l].type = CONV;
    } else if (strcmp(type, POOLING_TYPE) == 0) {
        cfg_t* pool_cfg = cfg_getsec(layer_opts, "pooling_param");
        const char* pool_type = cfg_getstr(pool_cfg, "pool");
        if (strcmp(pool_type, MAX_POOL_TYPE) == 0) {
            layers[l].type = POOL_MAX;
        } else if (strcmp(pool_type, AVG_POOL_TYPE) == 0) {
            layers[l].type = POOL_AVG;
        } else {
            assert(false && "Invalid type of pooling layer!");
        }
    } else if (strcmp(type, FC_TYPE) == 0) {
        layers[l].type = FC;
    } else {
        assert(false && "Invalid layer type!");
    }
}

static void set_layer_aux_params(layer_t* layers, cfg_t* layer_opts, int l) {
    if (layers[l].type == CONV) {
        cfg_t* conv_params = cfg_getsec(layer_opts, "convolution_param");
        layers[l].c_kernel_size = cfg_getint(conv_params, "kernel_size");
        layers[l].c_padding = cfg_getint(conv_params, "pad");
        layers[l].output_height = cfg_getint(conv_params, "num_output");
        assert(layers[l].c_kernel_size != -1);
        assert(layers[l].c_padding != -1);
        assert(layers[l].output_height != -1);
    } else if (layers[l].type == POOL_MAX) {
        cfg_t* pool_params = cfg_getsec(layer_opts, "pooling_param");
        layers[l].p_size = cfg_getint(pool_params, "size");
        layers[l].p_stride = cfg_getint(pool_params, "stride");
        assert(layers[l].p_size != -1);
        assert(layers[l].p_stride != -1);
    }
}

static void set_layer_input_dims(layer_t* layers, cfg_t* layer_opts, int l) {
    if (l == 0) {
        layers[l].input_rows = input_rows;
        layers[l].input_cols = input_cols;
        layers[l].input_height = input_height;
    } else {
        layers[l].input_rows = layers[l - 1].output_rows;
        layers[l].input_cols = layers[l - 1].output_cols;
        layers[l].input_height = layers[l - 1].output_height;
    }

    // Make some adjustments.
    if (layers[l].type == CONV) {
        // Add padding.
        layers[l].input_rows += layers[l].c_padding * 2;
        layers[l].input_cols += layers[l].c_padding * 2;
    } else if (layers[l].type == FC) {
        cfg_t* fc_params = cfg_getsec(layer_opts, "inner_product_param");
        if (l == 0 || (l > 0 && layers[l - 1].type != FC)) {
            // If this is the first layer, or this layer is a transition from a
            // CONV or POOL to a FC layer, we have to flatten the previous
            // layer's output.
            layers[l].input_rows = layers[l].input_rows *
                                   layers[l].input_cols *
                                   layers[l].input_height;
        } else {
            layers[l].input_rows = layers[l-1].output_cols;
        }
        layers[l].input_rows += 1;  // For the bias.
        // This is the number of hidden nodes in this layer.
        layers[l].input_cols = cfg_getint(fc_params, "num_output");
        layers[l].input_height = 1;
    } else if (layers[l].type == POOL_MAX) {
        // Nothing more to do.
    }
}

static void set_layer_output_dims(layer_t* layers, cfg_t* layer_opts, int l) {
    if (layers[l].type == CONV) {
        cfg_t* conv_params = cfg_getsec(layer_opts, "convolution_param");
        layers[l].output_rows = layers[l].input_rows - 2 * layers[l].c_padding;
        layers[l].output_cols = layers[l].input_cols - 2 * layers[l].c_padding;
        // Number of kernels is the third dimension of the output.
        layers[l].output_height = cfg_getint(conv_params, "num_output");
    } else if (layers[l].type == POOL_MAX) {
        layers[l].output_rows =
                (layers[l].input_rows - layers[l].p_size) / layers[l].p_stride +
                1;
        layers[l].output_cols =
                (layers[l].input_cols - layers[l].p_size) / layers[l].p_stride +
                1;
        layers[l].output_height = layers[l].input_height;
    } else if (layers[l].type == FC) {
        layers[l].output_cols = layers[l].input_cols;
        layers[l].output_rows = 1;
        layers[l].output_height = 1;
    }
}

static void read_top_level_config(layer_t* layers, cfg_t* network_opts) {
    input_rows = cfg_getint(network_opts, "input_rows");
    input_cols = cfg_getint(network_opts, "input_cols");
    input_height = cfg_getint(network_opts, "input_height");
}

static void read_layer_config(layer_t* layers, cfg_t* network_opts, int l) {
    cfg_t* current_layer_opts = cfg_getnsec(network_opts, "layer", l);
    set_layer_type(layers, current_layer_opts, l);
    set_layer_aux_params(layers, current_layer_opts, l);
    set_layer_input_dims(layers, current_layer_opts, l);
    set_layer_output_dims(layers, current_layer_opts, l);
}

static void print_layer_config(layer_t* layers, int num_layers) {
    printf("==================================\n");
    printf("Network configuration (per input):\n");
    printf("----------------------------------\n");
    for (int i = 0; i < num_layers; i++) {
        layer_type type = layers[i].type;
        printf("  Layer %d: ", i);
        if (type == CONV) {
            printf("  Convolutional\n");
            printf("    Input size: %d x %d x %d (after padding)\n",
                   layers[i].input_rows, layers[i].input_cols,
                   layers[i].input_height);
            printf("    Output size: %d x %d x %d\n", layers[i].output_rows,
                   layers[i].output_cols, layers[i].output_height);
            printf("    Kernel size: %d x %d x %d\n", layers[i].c_kernel_size,
                   layers[i].c_kernel_size, layers[i].input_height);
            printf("    Num kernels: %d\n", layers[i].output_height);
            printf("    Padding: %d\n", layers[i].c_padding);
        } else if (type == FC) {
            printf("  Fully connected\n");
            printf("    Weights: %d x %d\n", layers[i].input_rows,
                   layers[i].input_cols);
        } else if (type == POOL_MAX) {
            printf("  Max pooling\n");
            printf("    Input size: %d x %d x %d\n", layers[i].input_rows,
                   layers[i].input_cols, layers[i].input_height);
            printf("    Output size: %d x %d x %d\n", layers[i].output_rows,
                   layers[i].output_cols, layers[i].output_height);
            printf("    Field size: %d\n", layers[i].p_size);
            printf("    Stride: %d\n", layers[i].p_stride);
            printf("    Height: %d\n", layers[i].output_height);
        } else if (type == SOFTMAX) {
            printf("  Softmax\n");
        }
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
    int num_layers = cfg_size(network_opts, "layer");

    int err = posix_memalign(
            (void**)layers_ptr, CACHELINE_SIZE,
            next_multiple(sizeof(layer_t) * num_layers, CACHELINE_SIZE));
    ASSERT_MEMALIGN(layers, err);

    read_top_level_config(*layers_ptr, network_opts);

    for (int i = 0; i < num_layers; i++) {
        read_layer_config(*layers_ptr, network_opts, i);
    }

    print_layer_config(*layers_ptr, num_layers);
    return num_layers;
}
