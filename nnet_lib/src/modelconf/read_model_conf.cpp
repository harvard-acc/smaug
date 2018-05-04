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
const char HARD_TANH_TYPE[] = "HARD_TANH";
const char SIGMOID_TYPE[] = "SIGMOID";
const char SOFTMAX_TYPE[] = "SOFTMAX";
const char OFFLOAD_DMA[] = "DMA";
const char OFFLOAD_ACP[] = "ACP";
const char OFFLOAD_CACHE[] = "CACHE";
const char PADDING_SAME[] = "SAME";
const char PADDING_VALID[] = "VALID";
const char DMA_ALWAYS[] = "DMA_ALWAYS";
const char ACP_ALWAYS[] = "ACP_ALWAYS";
const char ACP_IF_WEIGHTS_REUSED[] = "ACP_IF_WEIGHTS_REUSED";

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

int validate_data_mvmt_policy(cfg_t* cfg, cfg_opt_t* opt) {
    const char* value = cfg_opt_getnstr(opt, cfg_opt_size(opt) - 1);
    assert(value);
    if (strcmp(value, DMA_ALWAYS) != 0 &&
        strcmp(value, ACP_ALWAYS) != 0 &&
        strcmp(value, ACP_IF_WEIGHTS_REUSED) != 0) {
        cfg_error(cfg, "Invalid data movement policy '%s' for '%s'!",
                  value, cfg->name);
        return -1;
    }
    return 0;
}

int validate_stride_params(cfg_t* cfg, cfg_opt_t* opt) {
    cfg_t* conv_params = cfg_opt_getnsec(opt, cfg_opt_size(opt) - 1);
    bool has_stride_set = cfg_size(conv_params, "stride");
    bool has_stride_rows_set = cfg_size(conv_params, "row_stride");
    bool has_stride_cols_set = cfg_size(conv_params, "col_stride");
    if (has_stride_set && (has_stride_rows_set || has_stride_cols_set)) {
        cfg_error(conv_params,
                  "If the stride parameter is set, then you cannot also set "
                  "the row and column strides independently!",
                  opt->name);
        return -1;
    } else if (!has_stride_set && (!has_stride_rows_set || !has_stride_cols_set)) {
        cfg_error(
                conv_params, "You must set both row and column strides!\n", opt->name);
        return -1;
    }
    if (has_stride_set) {
        int stride = cfg_getint(conv_params, "stride");
        cfg_setint(conv_params, "row_stride", stride);
        cfg_setint(conv_params, "col_stride", stride);
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
    return validate_stride_params(cfg, opt);
}

int validate_inner_product_params(cfg_t* cfg, cfg_opt_t* opt) {
    cfg_t* layer = cfg_opt_getnsec(opt, cfg_opt_size(opt) - 1);
    if (!cfg_size(layer, "num_output")) {
        cfg_error(cfg, "Missing required option 'num_output'!", opt->name);
        return -1;
    }
    return 0;
}

int validate_padding_type(cfg_t* cfg, cfg_opt_t* opt) {
    const char* pad_type = cfg_getstr(cfg, "padding");
    if (strncmp(pad_type, PADDING_SAME, 5) ||
        strncmp(pad_type, PADDING_VALID, 6)) {
        cfg_error(cfg, "Invalid padding type (options are SAME or VALID)!",
                  opt->name);
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
    // We either set kernel_size OR both kernel_cols and kernel_rows, but
    // not both.
    bool has_kernel_size_set = cfg_size(conv_params, "kernel_size");
    bool has_kernel_cols_set = cfg_size(conv_params, "kernel_cols");
    bool has_kernel_rows_set = cfg_size(conv_params, "kernel_rows");
    if (has_kernel_size_set && (has_kernel_cols_set || has_kernel_rows_set)) {
        cfg_error(conv_params,
                  "If kernel_size is set, then you cannot also specify "
                  "kernel_cols or kernel_rows!",
                  opt->name);
        return -1;
    } else if (!has_kernel_size_set &&
               (!has_kernel_cols_set || !has_kernel_rows_set)) {
        cfg_error(conv_params,
                  "You must set both kernel_cols and kernel_rows!",
                  opt->name);
        return -1;
    }
    if (has_kernel_size_set) {
        int kernel_size = cfg_getint(conv_params, "kernel_size");
        cfg_setint(conv_params, "kernel_cols", kernel_size);
        cfg_setint(conv_params, "kernel_rows", kernel_size);
    }
    return validate_stride_params(cfg, opt);
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
        strcmp(value, SIGMOID_TYPE) != 0 && strcmp(value, SOFTMAX_TYPE) != 0 &&
        strcmp(value, HARD_TANH_TYPE) != 0) {
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
    } else if (strcmp(activation, HARD_TANH_TYPE) == 0) {
        layers[l].activation = HARD_TANH;
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

static void set_layer_padding(layer_t* layers, int l, cfg_t* conv_params) {
    const char* pad_type = cfg_getstr(conv_params, "padding");
    if (strncmp(pad_type, PADDING_SAME, strlen(PADDING_SAME)) == 0) {
        int total_row_pad = layers[l].weights.rows - 1;
        int total_col_pad = layers[l].weights.cols - 1;
        padding pad;
        if (total_row_pad % 2 == 0) {
            pad.top = total_row_pad / 2;
            pad.bottom = pad.top;
        } else {
            pad.top = max2(total_row_pad / 2, 1);
            pad.bottom = total_row_pad - pad.top;
        }
        if (total_col_pad % 2 == 0) {
            pad.left = total_col_pad / 2;
            pad.right = pad.left;
        } else {
            pad.left = max2(total_col_pad / 2, 1);
            pad.right = total_col_pad - pad.left;
        }
        layers[l].pad = pad;
    } else if (strncmp(pad_type, PADDING_VALID, strlen(PADDING_SAME)) == 0) {
        layers[l].pad = (padding){ 0, 0, 0, 0 };
    } else {
        assert(false && "Unknown padding type!");
    }
}

static void set_layer_dims(layer_t* layers, cfg_t* layer_opts, int l) {
    if (layers[l].type == CONV_STANDARD) {
        layers[l].inputs.rows = layers[l - 1].outputs.rows;
        layers[l].inputs.cols = layers[l - 1].outputs.cols;
        layers[l].inputs.height = layers[l - 1].outputs.height;

        cfg_t* conv_params = cfg_getsec(layer_opts, "convolution_param");
        layers[l].weights.rows = cfg_getint(conv_params, "kernel_rows");
        layers[l].weights.cols = cfg_getint(conv_params, "kernel_cols");
        layers[l].weights.height = layers[l].inputs.height;
        layers[l].stride.rows = cfg_getint(conv_params, "row_stride");
        layers[l].stride.cols = cfg_getint(conv_params, "col_stride");
        layers[l].biases.rows = 0;
        layers[l].biases.cols = 0;
        layers[l].biases.height = 0;

        set_layer_padding(layers, l, conv_params);
#if ARCHITECTURE != EIGEN && ARCHITECTURE != SMV
        layers[l].inputs.rows += layers[l].pad.top + layers[l].pad.bottom;
        layers[l].inputs.cols += layers[l].pad.left + layers[l].pad.right;
        layers[l].outputs.rows = calc_conv_rows(&layers[l], false);
        layers[l].outputs.cols = calc_conv_cols(&layers[l], false);
#else
        layers[l].outputs.rows = calc_conv_rows(&layers[l], true);
        layers[l].outputs.cols = calc_conv_cols(&layers[l], true);
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
        layers[l].weights.rows = cfg_getint(conv_params, "kernel_rows");
        layers[l].weights.cols = cfg_getint(conv_params, "kernel_cols");
        layers[l].weights.height = 1;
        layers[l].stride.rows = cfg_getint(conv_params, "row_stride");
        layers[l].stride.cols = cfg_getint(conv_params, "col_stride");
        layers[l].biases.rows = 0;
        layers[l].biases.cols = 0;
        layers[l].biases.height = 0;

        set_layer_padding(layers, l, conv_params);
        layers[l].inputs.rows += layers[l].pad.top + layers[l].pad.bottom;
        layers[l].inputs.cols += layers[l].pad.left + layers[l].pad.right;
        layers[l].outputs.rows = calc_conv_rows(&layers[l], false);
        layers[l].outputs.cols = calc_conv_cols(&layers[l], false);
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
        layers[l].weights.rows = layers[l].inputs.height;
        layers[l].weights.cols = cfg_getint(conv_params, "num_output");
        layers[l].weights.height = 1;
        layers[l].biases.rows = 1;
        layers[l].biases.cols = cfg_getint(conv_params, "num_output");
        layers[l].biases.height = 1;
        layers[l].stride.rows = cfg_getint(conv_params, "row_stride");
        layers[l].stride.cols = cfg_getint(conv_params, "col_stride");

        layers[l].pad = (padding){ 0, 0, 0, 0 };
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

        layers[l].weights.rows = layers[l].inputs.cols;  // for bias.
        layers[l].weights.cols = cfg_getint(fc_params, "num_output");

        layers[l].biases.rows = 1;
        layers[l].biases.cols = cfg_getint(fc_params, "num_output");

        layers[l].outputs.rows = layers[l].inputs.rows;
        layers[l].outputs.cols = layers[l].weights.cols;

        layers[l].inputs.height = 1;
        layers[l].outputs.height = 1;
        layers[l].weights.height = 1;
        layers[l].biases.height = 1;
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
      layers[l].biases.rows = 0;
      layers[l].biases.cols = 0;
      layers[l].biases.height = 0;
      layers[l].stride.rows = cfg_getint(pool_params, "row_stride");
      layers[l].stride.cols = cfg_getint(pool_params, "col_stride");

      layers[l].outputs.rows = (layers[l].inputs.rows - layers[l].weights.cols) /
                                       layers[l].stride.rows +
                               1;
      layers[l].outputs.cols = (layers[l].inputs.cols - layers[l].weights.cols) /
                                       layers[l].stride.cols +
                               1;
      layers[l].outputs.height = layers[l].inputs.height;
      layers[l].pad = (padding){ 0, 0, 0, 0 };
      assert(layers[l].weights.rows != -1);
      return;
    }

    if (layers[l].type == BATCH_NORM) {
      layers[l].inputs.rows = layers[l - 1].outputs.rows;
      layers[l].inputs.cols = layers[l - 1].outputs.cols;
      layers[l].inputs.height = layers[l - 1].outputs.height;
      // We have to keep going backwards until we find the first non-batch
      // layer.
      int i = l - 1;
      while (i > 0 && layers[i].type == BATCH_NORM) {
          i--;
      }
      layer_type prev_layer_type = layers[i].type;

      // Rows are organized as {mean, var, gamma, beta}.
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
          case BATCH_NORM:
              cfg_error(layer_opts, "This batch norm layer has no preceding "
                                    "non-batch norm layer!");
              break;
          default:
              cfg_error(layer_opts, "Invalid location for batch norm layer.");
              break;
      }
      layers[l].biases.rows = 0;
      layers[l].biases.cols = 0;
      layers[l].biases.height = 0;
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
    if (TRANSPOSE_WEIGHTS == 1 && layers[l].type == FC) {
        // When FC weights are transposed (stored col-major), the dimension
        // that needs to be aligned now are the rows.
        layers[l].weights.align_pad =
                calc_padding(layers[l].weights.rows, data_alignment);
    } else {
        layers[l].weights.align_pad =
                calc_padding(layers[l].weights.cols, data_alignment);
    }
    layers[l].biases.align_pad =
            calc_padding(layers[l].biases.cols, data_alignment);
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
    layers[0].biases.rows = 0;
    layers[0].biases.cols = 0;
    layers[0].biases.height = 0;
    layers[0].num = 0;
    layers[0].host_weights = NULL;

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
    layers[l].host_weights = NULL;
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

data_mvmt_policy str2policy(const char* value) {
    if (strcmp(value, DMA_ALWAYS) == 0)
        return DmaAlways;
    if (strcmp(value, ACP_ALWAYS) == 0)
        return AcpAlways;
    if (strcmp(value, ACP_IF_WEIGHTS_REUSED) == 0)
        return AcpIfWeightsAreReused;
    return NumDataMvmtPolicies;
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
        device->use_hw_pooling = cfg_getbool(device_opts, "use_hw_pooling");
        device->use_pipelined_dma =
                cfg_getbool(device_opts, "use_pipelined_dma");
        device->use_pipelined_activation_func =
                cfg_getbool(device_opts, "use_pipelined_activation_func");
        device->umem_size = cfg_getint(device_opts, "umem_size");
        device->spad_size = cfg_getint(device_opts, "spad_size");
        device->l2_size =
                cfg_getint(device_opts, "l2_size");
        device->weights_load_policy =
                str2policy(cfg_getstr(device_opts, "weights_load_policy"));
    } else {
        device->cpu_default_offload = IO_DMA;
        device->cpu_pooling_offload = IO_DMA;
        device->cpu_activation_func_offload = IO_DMA;
        device->use_hw_activation_func = true;
        device->use_hw_batch_norm = true;
        device->use_hw_pooling = true;
        device->use_pipelined_dma = false;
        device->use_pipelined_activation_func = false;
        device->umem_size = 0;
        device->spad_size = 0;
        device->l2_size = 0;
        device->weights_load_policy = DmaAlways;
    }
}

static void read_sampling_param(cfg_t* all_opts, sampling_param_t* sampling) {
    if (cfg_size(all_opts, "sampling_param") != 0) {
        cfg_t* sampling_opts = cfg_getsec(all_opts, "sampling_param");
        sampling->standard_conv_num_filters =
                cfg_getint(sampling_opts, "standard_conv_num_filters");
        sampling->fc_num_neurons =
                cfg_getint(sampling_opts, "fc_num_neurons");
        sampling->smv_conv_inner_iters =
                cfg_getint(sampling_opts, "smv_conv_inner_iters");
        sampling->smv_conv_input_tiles =
                cfg_getint(sampling_opts, "smv_conv_input_tiles");
        sampling->smv_conv_output_tiles =
                cfg_getint(sampling_opts, "smv_conv_output_tiles");
        sampling->smv_conv_l2_tiles =
                cfg_getint(sampling_opts, "smv_conv_l2_tiles");
    } else {
        sampling->standard_conv_num_filters = 0;
        sampling->fc_num_neurons = 0;
        sampling->smv_conv_inner_iters = 0;
        sampling->smv_conv_input_tiles = 0;
        sampling->smv_conv_output_tiles = 0;
        sampling->smv_conv_l2_tiles = 0;
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
            printf("    Kernel size: %d x %d x %d\n", layers[i].weights.rows,
                   layers[i].weights.cols, layers[i].weights.height);
            printf("    Num kernels: %d\n", layers[i].outputs.height);
            printf("    Padding: %d\n", layers[i].pad.top);
            printf("    Row stride: %d\n", layers[i].stride.rows);
            printf("    Col stride: %d\n", layers[i].stride.cols);
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
            printf("    Row stride: %d\n", layers[i].stride.rows);
            printf("    Col stride: %d\n", layers[i].stride.cols);
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
            printf("    Row stride: %d\n", layers[i].stride.rows);
            printf("    Col stride: %d\n", layers[i].stride.cols);
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
               act == SOFTMAX ? "SOFTMAX" :
               act == HARD_TANH ? "HARD TANH ": "NONE");
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
           "   Use HW activation function: %s\n"
           "   Use HW batch norm: %s\n"
           "   Use HW pooling: %s\n"
           "   Use pipelined DMA: %s\n"
           "   Use pipelined activation function: %s\n",
           io_req_to_str(device->cpu_default_offload),
           io_req_to_str(device->cpu_pooling_offload),
           io_req_to_str(device->cpu_activation_func_offload),
           bool_to_yesno(device->use_hw_activation_func),
           bool_to_yesno(device->use_hw_batch_norm),
           bool_to_yesno(device->use_hw_pooling),
           bool_to_yesno(device->use_pipelined_dma),
           bool_to_yesno(device->use_pipelined_activation_func));
}

static void print_sampling_param(sampling_param_t* sampling_param) {
    printf("========================================\n");
    printf("Sampling configuration\n");
    printf("----------------------------------------\n"
           "   Standard convolution filters: %d\n"
           "   FC num neurons: %d\n"
           "   SMV convolution inner iters: %d\n"
           "   SMV convolution input tiles: %d\n"
           "   SMV convolution output tiles: %d\n"
           "   SMV convolution l2 tiles: %d\n",
           sampling_param->standard_conv_num_filters,
           sampling_param->fc_num_neurons,
           sampling_param->smv_conv_inner_iters,
           sampling_param->smv_conv_input_tiles,
           sampling_param->smv_conv_output_tiles,
           sampling_param->smv_conv_l2_tiles);
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
    cfg_set_validate_func(cfg, "network|layer|convolution_param|stride",
                          validate_unsigned_int);
    cfg_set_validate_func(cfg, "network|layer|convolution_param|row_stride",
                          validate_unsigned_int);
    cfg_set_validate_func(cfg, "network|layer|convolution_param|col_stride",
                          validate_unsigned_int);
    cfg_set_validate_func(cfg, "network|layer|convolution_param|kernel_size",
                          validate_unsigned_int);
    cfg_set_validate_func(cfg, "network|layer|convolution_param|kernel_rows",
                          validate_unsigned_int);
    cfg_set_validate_func(cfg, "network|layer|convolution_param|kernel_cols",
                          validate_unsigned_int);
    cfg_set_validate_func(
            cfg, "network|layer|convolution_param|pad", validate_padding_type);
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
    cfg_set_validate_func(
            cfg, "sampling_param|fc_num_neurons", validate_unsigned_int);
    cfg_set_validate_func(
            cfg, "sampling_param|smv_conv_inner_iters", validate_unsigned_int);
    cfg_set_validate_func(
            cfg, "sampling_param|smv_conv_input_tiles", validate_unsigned_int);
    cfg_set_validate_func(
            cfg, "sampling_param|smv_conv_output_tiles", validate_unsigned_int);
    cfg_set_validate_func(
            cfg, "sampling_param|smv_conv_l2_tiles", validate_unsigned_int);
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
    assert(!((*device_ptr)->use_pipelined_activation_func == true &&
             (*device_ptr)->use_hw_activation_func == true));

    // Read the sampling configuration.
    *sampling_ptr = (sampling_param_t*) malloc_aligned(sizeof(sampling_param_t));
    read_sampling_param(all_opts, *sampling_ptr);

    // Set some global variables.
    layer_t* last_layer = &layers[num_layers - 1];
    NUM_CLASSES = last_layer->outputs.rows * last_layer->outputs.cols *
                  last_layer->outputs.height;
    INPUT_DIM = input_rows * input_cols * input_height;

    assert(INPUT_DIM > 0);
    assert(NUM_CLASSES > 0);

    print_layer_config(*layers_ptr, num_layers);
    print_device_config(*device_ptr);
    print_sampling_param(*sampling_ptr);
    cfg_free(all_opts);
    return num_layers;
}
