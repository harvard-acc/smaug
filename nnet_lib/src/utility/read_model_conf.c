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

const char CONV_STANDARD_TYPE[] = "CONVOLUTION";
const char CONV_DEPTHWISE_TYPE[] = "DEPTHWISE_CONVOLUTION";
const char CONV_POINTWISE_TYPE[] = "POINTWISE_CONVOLUTION";
const char FC_TYPE[] = "INNER_PRODUCT";
const char POOLING_TYPE[] = "POOLING";
const char MAX_POOL_TYPE[] = "MAX";
const char AVG_POOL_TYPE[] = "AVG";
const char BATCH_NORM_TYPE[] = "BATCH_NORM";
const char NONE_TYPE[] = "NONE";
const char RELU_TYPE[] = "RELU";
const char LRELU_TYPE[] = "LRELU";
const char ELU_TYPE[] = "ELU";
const char SELU_TYPE[] = "SELU";
const char TANH_TYPE[] = "TANH";
const char SIGMOID_TYPE[] = "SIGMOID";
const char SOFTMAX_TYPE[] = "SOFTMAX";
const char OFFLOAD_DMA[] = "DMA";
const char OFFLOAD_ACP[] = "ACP";
const char OFFLOAD_CACHE[] = "CACHE";

static int input_rows;
static int input_cols;
static int input_height;
static int data_alignment;

int validate_network(cfg_t* cfg, cfg_opt_t* opt) {
    cfg_t* network = cfg_opt_getnsec(opt, cfg_opt_size(opt) - 1);
    if (!cfg_size(network, "input_rows")) {
        cfg_error(cfg, "Missing required option 'input_rows'!");
        return -1;
    }
    if (!cfg_size(network, "input_cols")) {
        cfg_error(cfg, "Missing required option 'input_cols'!");
        return -1;
    }
    if (!cfg_size(network, "input_height")) {
        cfg_error(cfg, "Missing required option 'input_height'!");
        return -1;
    }
    return 0;
}

int validate_offload_mechanism(cfg_t* cfg, cfg_opt_t* opt) {
    const char* value = cfg_opt_getnstr(opt, cfg_opt_size(opt) - 1);
    assert(value);
    if (strcmp(value, OFFLOAD_DMA) != 0 && strcmp(value, OFFLOAD_ACP) != 0 &&
        strcmp(value, OFFLOAD_CACHE) != 0) {
        cfg_error(cfg,
                  "'%s' is an invalid option for option '%s': Supported "
                  "options are DMA, ACP, or CACHE.",
                  value, opt->name);
        return -1;
    }
    return 0;
}

int validate_layer_section(cfg_t* cfg, cfg_opt_t* opt) {
    cfg_t* layer = cfg_opt_getnsec(opt, cfg_opt_size(opt) - 1);
    if (cfg_size(layer, "type") == 0) {
        cfg_error(cfg, "Missing required option 'type' in layer '%s'.",
                  cfg_title(layer));
        return -1;
    }
    if (!cfg_size(layer, "inner_product_param") &&
        !cfg_size(layer, "convolution_param") &&
        !cfg_size(layer, "pooling_param") &&
        // Batch norm layer does not have user-specified parameters
        strcmp(cfg_getstr(layer, "type"), BATCH_NORM_TYPE) != 0) {
        cfg_error(cfg, "Layer '%s' is missing layer-specific parameters!",
                  cfg_title(layer));
        return -1;
    }
    return 0;
}

int validate_layer_type(cfg_t* cfg, cfg_opt_t* opt) {
    const char* value = cfg_opt_getnstr(opt, cfg_opt_size(opt) - 1);
    assert(value);
    if (strcmp(value, CONV_STANDARD_TYPE) != 0 &&
        strcmp(value, CONV_DEPTHWISE_TYPE) != 0 &&
        strcmp(value, CONV_POINTWISE_TYPE) != 0 &&
        strcmp(value, FC_TYPE) != 0 &&
        strcmp(value, POOLING_TYPE) != 0 &&
        strcmp(value, BATCH_NORM_TYPE) != 0) {
        cfg_error(cfg, "Invalid layer type '%s' for '%s'!", value, cfg->name);
        return -1;
    }
    return 0;
}

int validate_pool_params(cfg_t* cfg, cfg_opt_t* opt) {
    cfg_t* layer = cfg_opt_getnsec(opt, cfg_opt_size(opt) - 1);
    if (!cfg_size(layer, "pool")) {
        cfg_error(cfg, "Missing required option 'pool'!", opt->name);
        return -1;
    }
    if (!cfg_size(layer, "size")) {
        cfg_error(cfg, "Missing required option 'size'!", opt->name);
        return -1;
    }
    if (!cfg_size(layer, "stride")) {
        cfg_error(cfg, "Missing required option 'stride'!", opt->name);
        return -1;
    }
    return 0;
}

int validate_inner_product_params(cfg_t* cfg, cfg_opt_t* opt) {
    cfg_t* layer = cfg_opt_getnsec(opt, cfg_opt_size(opt) - 1);
    if (!cfg_size(layer, "num_output")) {
        cfg_error(cfg, "Missing required option 'num_output'!", opt->name);
        return -1;
    }
    return 0;
}
int validate_conv_params(cfg_t* cfg, cfg_opt_t* opt) {
    cfg_t* conv_params = cfg_opt_getnsec(opt, cfg_opt_size(opt) - 1);
    const char* conv_type = cfg_getstr(cfg, "type");
    bool is_depthwise_conv = (strcmp(conv_type, CONV_DEPTHWISE_TYPE) == 0);
    if (!cfg_size(conv_params, "num_output")) {
        if (!is_depthwise_conv) {
            cfg_error(conv_params, "Missing required option 'num_output'!",
                      opt->name);
            return -1;
        }
    } else if (is_depthwise_conv) {
        cfg_error(
                conv_params,
                "Depthwise convolution layers do not need 'num_output' "
                "specified, as it is implied by the number of input channels!",
                opt->name);
        return -1;
    }
    if (!cfg_size(conv_params, "kernel_size")) {
        cfg_error(conv_params, "Missing required option 'kernel_size'!",
                  opt->name);
        return -1;
    }
    if (!cfg_size(conv_params, "stride")) {
        cfg_error(conv_params, "Missing required option 'stride'!", opt->name);
        return -1;
    }
    bool is_pointwise_conv = (strcmp(conv_type, CONV_POINTWISE_TYPE) == 0);
    if (!cfg_size(conv_params, "pad")) {
        if (!is_pointwise_conv) {
            cfg_error(conv_params, "Missing required option 'pad'!", opt->name);
            return -1;
        }
    } else if (is_pointwise_conv) {
        cfg_error(conv_params,
                  "Pointwise convolution layers do not accept a padding "
                  "parameter (implied padding is 0)!",
                  opt->name);
        return -1;
    }
    return 0;
}

int validate_pool_layer(cfg_t* cfg, cfg_opt_t* opt) {
    cfg_t* layer = cfg_opt_getnsec(opt, cfg_opt_size(opt) - 1);
    if (!cfg_size(layer, "pool")) {
        cfg_error(cfg, "Missing required option 'pool'!", opt->name);
        return -1;
    }
    if (!cfg_size(layer, "size")) {
        cfg_error(cfg, "Missing required option 'size'!", opt->name);
        return -1;
    }
    if (!cfg_size(layer, "stride")) {
        cfg_error(cfg, "Missing required option 'stride'!", opt->name);
        return -1;
    }
    return 0;
}

int validate_conv_type(cfg_t* cfg, cfg_opt_t* opt) {
    const char* value = cfg_opt_getnstr(opt, cfg_opt_size(opt) - 1);
    assert(value);
    if (strcmp(value, CONV_STANDARD_TYPE) != 0 &&
        strcmp(value, CONV_DEPTHWISE_TYPE) != 0) {
        cfg_error(cfg, "Invalid pooling type '%s'!", value);
        return -1;
    }
    return 0;
}

int validate_pool_type(cfg_t* cfg, cfg_opt_t* opt) {
    const char* value = cfg_opt_getnstr(opt, cfg_opt_size(opt) - 1);
    assert(value);
    if (strcmp(value, MAX_POOL_TYPE) != 0 &&
        strcmp(value, AVG_POOL_TYPE) != 0) {
        cfg_error(cfg, "Invalid pooling type '%s'!", value);
        return -1;
    }
    return 0;
}

int validate_activation_func(cfg_t* cfg, cfg_opt_t* opt) {
    const char* value = cfg_opt_getnstr(opt, cfg_opt_size(opt) - 1);
    assert(value);
    if (strcmp(value, NONE_TYPE) != 0 && strcmp(value, RELU_TYPE) != 0 &&
        strcmp(value, LRELU_TYPE) != 0 && strcmp(value, ELU_TYPE) != 0 &&
        strcmp(value, SELU_TYPE) != 0 && strcmp(value, TANH_TYPE) != 0 &&
        strcmp(value, SIGMOID_TYPE) != 0 && strcmp(value, SOFTMAX_TYPE) != 0) {
        cfg_error(cfg, "Invalid activation function '%s' for layer '%s'!",
                  value, cfg_title(cfg));
        return -1;
    }

    return 0;
}

int validate_unsigned_int(cfg_t* cfg, cfg_opt_t* opt) {
    int value = cfg_opt_getnint(opt, cfg_opt_size(opt) - 1);
    if (value < 0) {
        cfg_error(cfg, "'%s' in '%s' must be positive!", opt->name, cfg->name);
        return -1;
    }
    return 0;
}

static void set_layer_type(layer_t* layers, cfg_t* layer_opts, int l) {
    const char* type = cfg_getstr(layer_opts, "type");
    if (strcmp(type, CONV_STANDARD_TYPE) == 0) {
        layers[l].type = CONV_STANDARD;
    } else if (strcmp(type, CONV_DEPTHWISE_TYPE) == 0) {
        layers[l].type = CONV_DEPTHWISE;
    } else if (strcmp(type, CONV_POINTWISE_TYPE) == 0) {
        layers[l].type = CONV_POINTWISE;
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
    } else if (strcmp(type, BATCH_NORM_TYPE) == 0) {
        layers[l].type = BATCH_NORM;
    } else {
        assert(false && "Invalid layer type!");
    }

    const char* activation = cfg_getstr(layer_opts, "activation");
    if (strcmp(activation, RELU_TYPE) == 0) {
        layers[l].activation = RELU;
    } else if (strcmp(activation, LRELU_TYPE) == 0) {
        layers[l].activation = LRELU;
    } else if (strcmp(activation, ELU_TYPE) == 0) {
        layers[l].activation = ELU;
    } else if (strcmp(activation, SELU_TYPE) == 0) {
        layers[l].activation = SELU;
    } else if (strcmp(activation, TANH_TYPE) == 0) {
        layers[l].activation = TANH;
    } else if (strcmp(activation, SIGMOID_TYPE) == 0) {
        layers[l].activation = SIGMOID;
    } else if (strcmp(activation, SOFTMAX_TYPE) == 0) {
        layers[l].activation = SOFTMAX;
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
    if (layers[l].type == CONV_STANDARD) {
        layers[l].inputs.rows = layers[l - 1].outputs.rows;
        layers[l].inputs.cols = layers[l - 1].outputs.cols;
        layers[l].inputs.height = layers[l - 1].outputs.height;

        cfg_t* conv_params = cfg_getsec(layer_opts, "convolution_param");
        layers[l].weights.rows = cfg_getint(conv_params, "kernel_size");
        layers[l].weights.cols = cfg_getint(conv_params, "kernel_size");
        layers[l].weights.height = layers[l].inputs.height;
        layers[l].field_stride = cfg_getint(conv_params, "stride");

        layers[l].c_padding = cfg_getint(conv_params, "pad");
#if ARCHITECTURE != EIGEN
        layers[l].inputs.rows += layers[l].c_padding * 2;
        layers[l].inputs.cols += layers[l].c_padding * 2;
        layers[l].outputs.rows =
                (layers[l].inputs.rows - layers[l].weights.cols) /
                        layers[l].field_stride + 1;
        layers[l].outputs.cols =
                (layers[l].inputs.cols - layers[l].weights.cols) /
                        layers[l].field_stride + 1;
#else
        layers[l].outputs.rows =
                (layers[l].inputs.rows - layers[l].weights.cols +
                 2 * layers[l].c_padding) /
                        layers[l].field_stride + 1;
        layers[l].outputs.cols =
                (layers[l].inputs.cols - layers[l].weights.cols +
                 2 * layers[l].c_padding) /
                        layers[l].field_stride + 1;

#endif

        // Number of kernels is the third dimension of the output.
        layers[l].outputs.height = cfg_getint(conv_params, "num_output");

        assert(layers[l].outputs.rows > 0);
        assert(layers[l].outputs.cols > 0);
        return;
    }

    if (layers[l].type == CONV_DEPTHWISE) {
        layers[l].inputs.rows = layers[l - 1].outputs.rows;
        layers[l].inputs.cols = layers[l - 1].outputs.cols;
        layers[l].inputs.height = layers[l - 1].outputs.height;

        cfg_t* conv_params = cfg_getsec(layer_opts, "convolution_param");
        layers[l].weights.rows = cfg_getint(conv_params, "kernel_size");
        layers[l].weights.cols = cfg_getint(conv_params, "kernel_size");
        layers[l].weights.height = 1;
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
        layers[l].outputs.height = layers[l].inputs.height;

        assert(layers[l].outputs.rows > 0);
        assert(layers[l].outputs.cols > 0);
        return;
    }

    if (layers[l].type == CONV_POINTWISE) {
        layers[l].inputs.rows = layers[l - 1].outputs.rows;
        layers[l].inputs.cols = layers[l - 1].outputs.cols;
        layers[l].inputs.height = layers[l - 1].outputs.height;

        // 1x1 convolutions use FC-formatted weights, so each row is a filter and
        // each col maps to a channel in the input.
        cfg_t* conv_params = cfg_getsec(layer_opts, "convolution_param");
        layers[l].weights.rows = layers[l].inputs.height + 1;
        layers[l].weights.cols = cfg_getint(conv_params, "num_output");
        layers[l].weights.height = 1;
        layers[l].field_stride = cfg_getint(conv_params, "stride");

        layers[l].c_padding = 0;
        layers[l].outputs.rows = layers[l].inputs.rows;
        layers[l].outputs.cols = layers[l].inputs.cols;
        layers[l].outputs.height = cfg_getint(conv_params, "num_output");
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
      layers[l].c_padding = 0;
      assert(layers[l].weights.rows != -1);
      assert(layers[l].field_stride != -1);
      return;
    }

    if (layers[l].type == BATCH_NORM) {
      layers[l].inputs.rows = layers[l - 1].outputs.rows;
      layers[l].inputs.cols = layers[l - 1].outputs.cols;
      layers[l].inputs.height = layers[l - 1].outputs.height;
      // Rows are organized as {mean, var, gamma, beta}.
      layer_type prev_layer_type = layers[l - 1].type;
      switch (prev_layer_type) {
          case CONV_STANDARD:
          case CONV_DEPTHWISE:
          case CONV_POINTWISE:
          case INPUT:
          case POOLING:
              layers[l].weights.rows = 4;
              layers[l].weights.cols = layers[l].inputs.height;
              layers[l].weights.height = 1;
              break;
          case FC:
              layers[l].weights.rows = layers[l].inputs.rows * 4;
              layers[l].weights.cols = layers[l].inputs.cols;
              layers[l].weights.height = layers[l].inputs.height;
              break;
          default:
              cfg_error(layer_opts, "Invalid location for batch norm layer.");
              break;
      }
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
    layers[0].num = 0;

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

io_req_t str_to_io_req(char* value) {
    if (strncmp(value, OFFLOAD_DMA, 3) == 0)
      return IO_DMA;
    if (strncmp(value, OFFLOAD_ACP, 3) == 0)
      return IO_ACP;
    if (strncmp(value, OFFLOAD_CACHE, 5) == 0)
      return IO_CACHE;
    assert(false && "Invalid string value of an io_req_t!");
    return IO_NONE;
}

const char* io_req_to_str(io_req_t value) {
    switch (value) {
      case IO_DMA: return OFFLOAD_DMA;
      case IO_ACP: return OFFLOAD_ACP;
      case IO_CACHE: return OFFLOAD_CACHE;
      default:
          assert(false && "Invalid string value of an io_req_t!");
          break;
    }
    return NONE_TYPE;
}

static void read_device_parameters(cfg_t* all_opts, device_t* device) {
    if (cfg_size(all_opts, "device") != 0) {
        cfg_t* device_opts = cfg_getsec(all_opts, "device");
        device->cpu_default_offload =
                str_to_io_req(cfg_getstr(device_opts, "cpu_default_offload"));
        device->cpu_pooling_offload =
                str_to_io_req(cfg_getstr(device_opts, "cpu_pooling_offload"));
        device->cpu_activation_func_offload = str_to_io_req(
                cfg_getstr(device_opts, "cpu_activation_func_offload"));
        device->use_hw_activation_func =
                cfg_getbool(device_opts, "use_hw_activation_func");
        device->use_hw_batch_norm =
                cfg_getbool(device_opts, "use_hw_batch_norm");
    } else {
        device->cpu_default_offload = IO_DMA;
        device->cpu_pooling_offload = IO_DMA;
        device->cpu_activation_func_offload = IO_DMA;
        device->use_hw_activation_func = true;
        device->use_hw_batch_norm = true;
    }
}

static void read_sampling_param(cfg_t* all_opts, sampling_param_t* sampling) {
    if (cfg_size(all_opts, "sampling_param") != 0) {
        cfg_t* sampling_opts = cfg_getsec(all_opts, "sampling_param");
        sampling->standard_conv_num_filters =
                cfg_getint(sampling_opts, "standard_conv_num_filters");
        sampling->fc_num_neurons =
                cfg_getint(sampling_opts, "fc_num_neurons");
    } else {
        sampling->standard_conv_num_filters = 0;
        sampling->fc_num_neurons = 0;
    }
}

static void print_layer_config(layer_t* layers, int num_layers) {
    printf("==================================\n");
    printf("Network configuration (per input):\n");
    printf("----------------------------------\n");
    for (int i = 0; i < num_layers; i++) {
        layer_type type = layers[i].type;
        activation_type act = layers[i].activation;
        printf("  Layer %d: ", i);
        if (type == CONV_STANDARD || type == CONV_DEPTHWISE) {
            if (type == CONV_STANDARD)
                printf("  Standard convolution\n");
            else
                printf("  Depthwise convolution\n");
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
        } else if (type == CONV_POINTWISE) {
            printf("  Pointwise convolution\n");
            printf("    Input size: %d x %d x %d (after padding)\n",
                   layers[i].inputs.rows, layers[i].inputs.cols,
                   layers[i].inputs.height);
            printf("    Output size: %d x %d x %d\n", layers[i].outputs.rows,
                   layers[i].outputs.cols, layers[i].outputs.height);
            printf("    Kernel size: 1 x 1 x %d\n", layers[i].inputs.height);
            printf("    Num kernels: %d\n", layers[i].outputs.height);
            printf("    Padding: 0\n");
            printf("    Stride: %d\n", layers[i].field_stride);
        } else if (type == FC) {
            printf("  Fully connected\n");
            printf("    Input size: %d x %d\n", layers[i].inputs.rows,
                   layers[i].inputs.cols);
            printf("    Weights: %d x %d\n", layers[i].weights.rows,
                   layers[i].weights.cols);
        } else if (type == POOLING) {
            if (layers[i].pool == MAX)
                printf("  Max pooling\n");
            else if (layers[i].pool == AVG)
                printf("  Average pooling\n");
            else
                assert(false && "Unknown pooling layer type!");
            printf("    Input size: %d x %d x %d\n", layers[i].inputs.rows,
                   layers[i].inputs.cols, layers[i].inputs.height);
            printf("    Output size: %d x %d x %d\n", layers[i].outputs.rows,
                   layers[i].outputs.cols, layers[i].outputs.height);
            printf("    Field size: %d\n", layers[i].weights.cols);
            printf("    Stride: %d\n", layers[i].field_stride);
            printf("    Height: %d\n", layers[i].outputs.height);
        } else if (type == BATCH_NORM) {
            printf("  Batch normalization\n");
            printf("    Input size: %d x %d x %d\n", layers[i].inputs.rows,
                   layers[i].inputs.cols, layers[i].inputs.height);
            printf("    Weight size: %d x %d x %d\n", layers[i].weights.rows,
                   layers[i].weights.cols, layers[i].weights.height);
            printf("    Output size: %d x %d x %d\n", layers[i].outputs.rows,
                   layers[i].outputs.cols, layers[i].outputs.height);
        } else if (type == INPUT) {
            printf("  Input layer\n");
            printf("    Input size: %d x %d x %d\n", layers[i].inputs.rows,
                   layers[i].inputs.cols, layers[i].inputs.height);
        }
        printf("    Input data padding: %d\n", layers[i].inputs.align_pad);
        printf("    Weight data padding: %d\n", layers[i].weights.align_pad);
        printf("    Output data padding: %d\n", layers[i].outputs.align_pad);
        printf("    Activation: %s\n",
               act == RELU ? "RELU" : act == SIGMOID ? "SIGMOID" :
               act == LRELU ? "LRELU" : act == ELU ? "ELU" :
               act == SELU ? "SELU" : act == TANH ? "TANH" :
               act == SOFTMAX ? "SOFTMAX" : "NONE");
    }
}

static void print_device_config(device_t* device) {
    printf("========================================\n");
    printf("Device configuration\n");
    printf("----------------------------------\n");

    printf("CPU offload mechanisms:\n"
           "   Default: %s\n"
           "   Pooling: %s\n"
           "   Activation function: %s\n"
           "   Use HW activation function: %s\n",
           io_req_to_str(device->cpu_default_offload),
           io_req_to_str(device->cpu_pooling_offload),
           io_req_to_str(device->cpu_activation_func_offload),
           device->use_hw_activation_func ? "yes" : "no");
}

static void print_sampling_param(sampling_param_t* sampling_param) {
    printf("========================================\n");
    printf("Sampling configuration\n");
    printf("----------------------------------------\n"
           "   Standard convolution filters: %d\n"
           "   FC num neurons: %d\n",
           sampling_param->standard_conv_num_filters,
           sampling_param->fc_num_neurons);
    printf("========================================\n");

}

static void install_validation_callbacks(cfg_t* cfg) {
    cfg_set_validate_func(cfg, "network", validate_network);
    cfg_set_validate_func(cfg, "network|layer", validate_layer_section);
    cfg_set_validate_func(cfg, "network|layer|type", validate_layer_type);
    cfg_set_validate_func(
            cfg, "network|layer|activation", validate_activation_func);
    cfg_set_validate_func(cfg, "network|input_rows", validate_unsigned_int);
    cfg_set_validate_func(cfg, "network|input_cols", validate_unsigned_int);
    cfg_set_validate_func(cfg, "network|input_height", validate_unsigned_int);
    cfg_set_validate_func(cfg, "network|layer|inner_product_param|num_output",
                          validate_unsigned_int);
    cfg_set_validate_func(cfg, "network|layer|convolution_param|type",
                          validate_conv_type);
    cfg_set_validate_func(cfg, "network|layer|convolution_param|num_output",
                          validate_unsigned_int);
    cfg_set_validate_func(cfg, "network|layer|convolution_param|kernel_size",
                          validate_unsigned_int);
    cfg_set_validate_func(cfg, "network|layer|convolution_param|stride",
                          validate_unsigned_int);
    cfg_set_validate_func(
            cfg, "network|layer|convolution_param|pad", validate_unsigned_int);
    cfg_set_validate_func(
            cfg, "network|layer|pooling_param|stride", validate_unsigned_int);
    cfg_set_validate_func(
            cfg, "network|layer|pooling_param|size", validate_unsigned_int);
    cfg_set_validate_func(
            cfg, "network|layer|pooling_param|pool", validate_pool_type);
    cfg_set_validate_func(
            cfg, "network|layer|pooling_param", validate_pool_params);
    cfg_set_validate_func(
            cfg, "network|layer|convolution_param", validate_conv_params);
    cfg_set_validate_func(cfg, "network|layer|inner_product_param",
                          validate_inner_product_params);
    cfg_set_validate_func(
            cfg, "device|cpu_default_offload", validate_offload_mechanism);
    cfg_set_validate_func(
            cfg, "device|cpu_pooling_offload", validate_offload_mechanism);
    cfg_set_validate_func(cfg, "device|cpu_activation_func_offload",
                          validate_offload_mechanism);
    cfg_set_validate_func(cfg,
                          "sampling_param|standard_conv_num_filters",
                          validate_unsigned_int);
    cfg_set_validate_func(cfg,
                          "sampling_param|fc_num_neurons",
                          validate_unsigned_int);
}

int configure_network_from_file(const char* cfg_file,
                                layer_t** layers_ptr,
                                device_t** device_ptr,
                                sampling_param_t** sampling_ptr) {
    cfg_t* all_opts = cfg_init(top_level_cfg, CFGF_NONE);
    install_validation_callbacks(all_opts);

    int ret = cfg_parse(all_opts, cfg_file);
    if (ret == CFG_FILE_ERROR) {
        assert(false && "Failed to open configuration file!");
    } else if (ret == CFG_PARSE_ERROR) {
        fprintf(stderr,
                "An error occurred when reading the configuration file!\n");
        exit(-1);
    }

    cfg_t* network_opts = cfg_getsec(all_opts, "network");
    int num_layers =
            cfg_size(network_opts, "layer") + 1;  // +1 for input layer.

    *layers_ptr = (layer_t*)malloc_aligned(sizeof(layer_t) * num_layers);
    layer_t* layers = *layers_ptr;

    //=---------------------  STEP 1 -----------------------=//
    // First, read in all the parameters from the configuration
    // file for each layer.

    read_top_level_config(layers, network_opts);
    for (int i = 1; i < num_layers; i++) {
        read_layer_config(layers, network_opts, i);
        layers[i].num = i;
    }

    //=---------------------  STEP 2 -----------------------=//
    // Identify layers that require their input to be flattened
    // (CONV/INPUT to FC) or unflattened (FC to CONV).

    layers[0].input_preprocessing = NO_PREPROCESSING;
    for (int i = 1; i < num_layers; i++) {
        if (layers[i].type == FC && layers[i-1].type != FC) {
            layers[i].input_preprocessing = FLATTEN;
        } else if ((layers[i].type == CONV_STANDARD ||
                    layers[i].type == CONV_DEPTHWISE ||
                    layers[i].type == CONV_POINTWISE) &&
                   layers[i - 1].type == FC) {
            layers[i].input_preprocessing = UNFLATTEN;
#if ARCHITECTURE == SMIV
        } else if (layers[i].type == CONV_POINTWISE &&
                   (layers[i - 1].type == CONV_STANDARD ||
                    layers[i - 1].type == CONV_DEPTHWISE ||
                    layers[i - 1].type == CONV_POINTWISE)) {
            layers[i].input_preprocessing = NCHW_TO_NHWC;
#endif
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

    // Read the device parameters.
    *device_ptr = (device_t*) malloc_aligned(sizeof(device_t));
    read_device_parameters(all_opts, *device_ptr);

    // Read the sampling configuration.
    *sampling_ptr = (sampling_param_t*) malloc_aligned(sizeof(sampling_param_t));
    read_sampling_param(all_opts, *sampling_ptr);

    // Set some global variables.
    NUM_CLASSES = layers[num_layers-1].outputs.cols;
    INPUT_DIM = input_rows * input_cols * input_height;

    assert(INPUT_DIM > 0);
    assert(NUM_CLASSES > 0);

    print_layer_config(*layers_ptr, num_layers);
    print_device_config(*device_ptr);
    print_sampling_param(*sampling_ptr);
    cfg_free(all_opts);
    return num_layers;
}
