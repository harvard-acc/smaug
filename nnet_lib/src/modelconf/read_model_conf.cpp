#include <cassert>
#include <cstdio>
#include <cstring>

#include <string>

#include "confuse.h"

#include "utility/utility.h"
#include "nnet_fwd.h"

#include "core/globals.h"
#include "core/tensor.h"
#include "core/network.h"
#include "core/workspace.h"
#include "modelconf/read_model_conf.h"
#include "operators/batch_norm_op.h"
#include "operators/convolution_op.h"
#include "operators/data_op.h"
#include "operators/depthwise_convolution_op.h"
#include "operators/eltwise_add_op.h"
#include "operators/elu_op.h"
#include "operators/inner_product_op.h"
#include "operators/pooling_op.h"
#include "operators/relu_op.h"
#include "operators/reorder_op.h"
#include "operators/sigmoid_op.h"
#include "operators/softmax_op.h"
#include "operators/tanh_op.h"
#include "utility/utils.h"

using namespace smaug;

extern cfg_opt_t convolution_param_cfg[];
extern cfg_opt_t inner_product_param_cfg[];
extern cfg_opt_t pooling_param_cfg[];
extern cfg_opt_t layer_cfg[];
extern cfg_opt_t network_cfg[];
extern cfg_opt_t top_level_cfg[];

static const std::string CONV_STANDARD_TYPE = "CONVOLUTION";
static const std::string CONV_DEPTHWISE_TYPE = "DEPTHWISE_CONVOLUTION";
static const std::string CONV_POINTWISE_TYPE = "POINTWISE_CONVOLUTION";
static const std::string FC_TYPE = "INNER_PRODUCT";
static const std::string POOLING_TYPE = "POOLING";
static const std::string MAX_POOL_TYPE = "MAX";
static const std::string AVG_POOL_TYPE = "AVG";
static const std::string BATCH_NORM_TYPE = "BATCH_NORM";
static const std::string FLATTEN_TYPE = "FLATTEN";
static const std::string NONE_TYPE = "NONE";
static const std::string ADD_TYPE = "ADD";
static const std::string RELU_TYPE = "RELU";
static const std::string LRELU_TYPE = "LRELU";
static const std::string ELU_TYPE = "ELU";
static const std::string SELU_TYPE = "SELU";
static const std::string TANH_TYPE = "TANH";
static const std::string HARD_TANH_TYPE = "HARD_TANH";
static const std::string SIGMOID_TYPE = "SIGMOID";
static const std::string SOFTMAX_TYPE = "SOFTMAX";
static const std::string OFFLOAD_DMA = "DMA";
static const std::string OFFLOAD_ACP = "ACP";
static const std::string OFFLOAD_CACHE = "CACHE";
static const std::string PADDING_SAME = "SAME";
static const std::string PADDING_VALID = "VALID";
static const std::string DMA_ALWAYS = "DMA_ALWAYS";
static const std::string ACP_ALWAYS = "ACP_ALWAYS";
static const std::string ACP_IF_WEIGHTS_REUSED = "ACP_IF_WEIGHTS_REUSED";

// TODO: Delete these.
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
    const char* op_str = cfg_opt_getnstr(opt, cfg_opt_size(opt) - 1);
    assert(op_str);
    std::string value(op_str);
    if (value != OFFLOAD_DMA && value != OFFLOAD_ACP &&
        value != OFFLOAD_CACHE) {
        cfg_error(cfg,
                  "'%s' is an invalid option for option '%s': Supported "
                  "options are DMA, ACP, or CACHE.",
                  value.c_str(), opt->name);
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
    // Some layer types do not have user-specified parameters.
    std::string layerType = cfg_getstr(layer, "type");
    if (layerType == FLATTEN_TYPE || layerType == BATCH_NORM_TYPE ||
        layerType == SOFTMAX_TYPE || layerType == ADD_TYPE) {
        return 0;
    }
    if (!cfg_size(layer, "inner_product_param") &&
        !cfg_size(layer, "convolution_param") &&
        !cfg_size(layer, "pooling_param")) {
        cfg_error(cfg, "Layer '%s' is missing layer-specific parameters!",
                  cfg_title(layer));
        return -1;
    }
    return 0;
}

int validate_layer_type(cfg_t* cfg, cfg_opt_t* opt) {
    const char* op_str = cfg_opt_getnstr(opt, cfg_opt_size(opt) - 1);
    assert(op_str);
    std::string value(op_str);
    if (value != CONV_STANDARD_TYPE && value != CONV_DEPTHWISE_TYPE &&
        value != CONV_POINTWISE_TYPE && value != FC_TYPE &&
        value != POOLING_TYPE && value != BATCH_NORM_TYPE &&
        value != FLATTEN_TYPE && value != ADD_TYPE) {
        cfg_error(cfg, "Invalid layer type '%s' for '%s'!", value.c_str(),
                  cfg->name);
        return -1;
    }
    return 0;
}

int validate_data_mvmt_policy(cfg_t* cfg, cfg_opt_t* opt) {
    const char* op_str = cfg_opt_getnstr(opt, cfg_opt_size(opt) - 1);
    assert(op_str);
    std::string value(op_str);
    if (value != DMA_ALWAYS && value != ACP_ALWAYS &&
        value != ACP_IF_WEIGHTS_REUSED) {
        cfg_error(cfg, "Invalid data movement policy '%s' for '%s'!",
                  value.c_str(), cfg->name);
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
    std::string pad_type = cfg_getstr(cfg, "padding");
    if (pad_type != PADDING_SAME && pad_type != PADDING_VALID) {
        cfg_error(cfg, "Invalid padding type (options are SAME or VALID)!",
                  opt->name);
        return -1;
    }
    return 0;
}

int validate_conv_params(cfg_t* cfg, cfg_opt_t* opt) {
    cfg_t* conv_params = cfg_opt_getnsec(opt, cfg_opt_size(opt) - 1);
    std::string conv_type = cfg_getstr(cfg, "type");
    bool is_depthwise_conv = (conv_type == CONV_DEPTHWISE_TYPE);
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
    const char* op_str = cfg_opt_getnstr(opt, cfg_opt_size(opt) - 1);
    assert(op_str);
    std::string value(op_str);
    if (value != CONV_STANDARD_TYPE && value != CONV_DEPTHWISE_TYPE) {
        cfg_error(cfg, "Invalid pooling type '%s'!", value.c_str());
        return -1;
    }
    return 0;
}

int validate_pool_type(cfg_t* cfg, cfg_opt_t* opt) {
    const char* op_str = cfg_opt_getnstr(opt, cfg_opt_size(opt) - 1);
    assert(op_str);
    std::string value(op_str);
    if (value != MAX_POOL_TYPE && value != AVG_POOL_TYPE) {
        cfg_error(cfg, "Invalid pooling type '%s'!", value.c_str());
        return -1;
    }
    return 0;
}

int validate_activation_func(cfg_t* cfg, cfg_opt_t* opt) {
    const char* op_str = cfg_opt_getnstr(opt, cfg_opt_size(opt) - 1);
    assert(op_str);
    std::string value(op_str);
    if (value != NONE_TYPE && value != RELU_TYPE && value != LRELU_TYPE &&
        value != ELU_TYPE && value != SELU_TYPE && value != TANH_TYPE &&
        value != SIGMOID_TYPE && value != SOFTMAX_TYPE &&
        value != HARD_TANH_TYPE) {
        cfg_error(cfg, "Invalid activation function '%s' for layer '%s'!",
                  value.c_str(), cfg_title(cfg));
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

static void createAndAddOperator(
        const std::string& name,
        const std::string& type,
        Network* network,
        Workspace* workspace,
        cfg_t* opCfg,
        std::vector<Operator*> inputs = std::vector<Operator*>()) {
    // If no input layers are specified, then it is assumed to be the last
    // operator that was added.
    if (inputs.empty())
        inputs.push_back(network->getLastOperator());
    if (type == CONV_STANDARD_TYPE) {
        cfg_t* convParams = cfg_getsec(opCfg, "convolution_param");
        ConvolutionOp<GlobalBackend>* op =
                new ConvolutionOp<GlobalBackend>(name, workspace);
        op->setWeightDims(cfg_getint(convParams, "kernel_rows"),
                          cfg_getint(convParams, "kernel_cols"),
                          cfg_getint(convParams, "num_output"));
        op->setStride(cfg_getint(convParams, "row_stride"),
                      cfg_getint(convParams, "col_stride"));
        op->setPadding(cfg_getstr(convParams, "padding"));
        network->addOperator(op, inputs);
    } else if (type == CONV_DEPTHWISE_TYPE) {
        cfg_t* convParams = cfg_getsec(opCfg, "convolution_param");
        DepthwiseConvolutionOp<GlobalBackend>* op =
                new DepthwiseConvolutionOp<GlobalBackend>(name, workspace);
        op->setWeightDims(cfg_getint(convParams, "kernel_rows"),
                          cfg_getint(convParams, "kernel_cols"),
                          cfg_getint(convParams, "num_output"));
        op->setStride(cfg_getint(convParams, "row_stride"),
                      cfg_getint(convParams, "col_stride"));
        op->setPadding(cfg_getstr(convParams, "padding"));
        network->addOperator(op, inputs);
    } else if (type == CONV_POINTWISE_TYPE) {
        assert(false && "Deprecated! Use normal convolution.");
    } else if (type == POOLING_TYPE) {
        cfg_t* poolCfg = cfg_getsec(opCfg, "pooling_param");
        std::string poolingType = cfg_getstr(poolCfg, "pool");
        PoolingOp<GlobalBackend>* op;
        if (poolingType == MAX_POOL_TYPE) {
            op = new MaxPoolingOp<GlobalBackend>(name, workspace);
        } else if (poolingType == AVG_POOL_TYPE) {
            op = new AvgPoolingOp<GlobalBackend>(name, workspace);
        } else {
            assert(false && "Invalid type of pooling layer!");
        }
        cfg_t* poolParams = cfg_getsec(opCfg, "pooling_param");
        op->setPoolingSize(cfg_getint(poolParams, "size"));
        op->setPoolingStride(cfg_getint(poolParams, "row_stride"),
                             cfg_getint(poolParams, "col_stride"));
        network->addOperator(op, inputs);
    } else if (type == FC_TYPE) {
        cfg_t* fcParams = cfg_getsec(opCfg, "inner_product_param");
        InnerProductOp<GlobalBackend>* op =
                new InnerProductOp<GlobalBackend>(name, workspace);
        op->setNumOutputs(cfg_getint(fcParams, "num_output"));
        network->addOperator(op, inputs);
        // TODO: This does not include the bias! Add an elementwise add
        // operation.
    } else if (type == FLATTEN_TYPE) {
        FlattenOp<GlobalBackend>* op =
                new FlattenOp<GlobalBackend>(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == BATCH_NORM_TYPE) {
        BatchNormOp<GlobalBackend>* op =
                new BatchNormOp<GlobalBackend>(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == ADD_TYPE) {
        EltwiseAddOp<GlobalBackend>* op =
                new EltwiseAddOp<GlobalBackend>(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == RELU_TYPE) {
        ReluOp<GlobalBackend>* op = new ReluOp<GlobalBackend>(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == LRELU_TYPE) {
        // TODO: Add parameter to enable customization of this behavior.
        ReluOp<GlobalBackend>* op =
                new ReluOp<GlobalBackend>(name, workspace, 0.1);
        network->addOperator(op, inputs);
    } else if (type == ELU_TYPE) {
        EluOp<GlobalBackend>* op = new EluOp<GlobalBackend>(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == SELU_TYPE) {
        SeluOp<GlobalBackend>* op = new SeluOp<GlobalBackend>(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == SIGMOID_TYPE) {
        SigmoidOp<GlobalBackend>* op =
                new SigmoidOp<GlobalBackend>(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == TANH_TYPE) {
        TanhOp<GlobalBackend>* op = new TanhOp<GlobalBackend>(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == HARD_TANH_TYPE) {
        HardTanhOp<GlobalBackend>* op =
                new HardTanhOp<GlobalBackend>(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == NONE_TYPE) {
        return;
    } else {
        assert(false && "Invalid layer type!");
    }
}

static void set_layer_padding(layer_t* layers, int l, cfg_t* conv_params) {
    std::string pad_type = cfg_getstr(conv_params, "padding");
    if (pad_type == PADDING_SAME) {
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
    } else if (pad_type == PADDING_VALID) {
        layers[l].pad = (padding){ 0, 0, 0, 0 };
    } else {
        assert(false && "Unknown padding type!");
    }
}

#if 0

io_req_t str_to_io_req(char* value) {
    if (value == OFFLOAD_DMA)
      return IO_DMA;
    if (value == OFFLOAD_ACP)
      return IO_ACP;
    if (value == OFFLOAD_CACHE)
      return IO_CACHE;
    assert(false && "Invalid string value of an io_req_t!");
    return IO_NONE;
}

const char* io_req_to_str(io_req_t value) {
    switch (value) {
      case IO_DMA: return OFFLOAD_DMA.c_str();
      case IO_ACP: return OFFLOAD_ACP.c_str();
      case IO_CACHE: return OFFLOAD_CACHE.c_str();
      default:
          assert(false && "Invalid string value of an io_req_t!");
          break;
    }
    return NONE_TYPE.c_str();
}

data_mvmt_policy str2policy(const char* value) {
    if (value == DMA_ALWAYS)
        return DmaAlways;
    if (value == ACP_ALWAYS)
        return AcpAlways;
    if (value == ACP_IF_WEIGHTS_REUSED)
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

#endif

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

static void readInputDataConfig(Network* network,
                                Workspace* workspace,
                                cfg_t* network_opts) {
    int height = cfg_getint(network_opts, "input_height");
    int rows = cfg_getint(network_opts, "input_rows");
    int cols = cfg_getint(network_opts, "input_cols");
    // TODO: Add a config parameter to specify the data layout of the model.
    // For now, assume NCHW format.
    DataLayout layout = DataLayout::NCHW;
    TensorShape shape({ 1, height, rows, cols }, layout);
    Tensor<GlobalBackend>* inputData =
            new Tensor<GlobalBackend>("input", shape);
    DataOp<GlobalBackend>* inputOp =
            new DataOp<GlobalBackend>("input", workspace);
    inputOp->setData(inputData);
    workspace->addTensor(inputData);
    network->addOperator(inputOp);

    // Set the global variables.
    // TODO: Remove.
    data_alignment = DATA_ALIGNMENT;
}

// TODO: Read the device/sampling parameters.
Network* smaug::readModelConfiguration(const std::string& cfg_file,
                                       Workspace* workspace) {
    cfg_t* all_opts = cfg_init(top_level_cfg, CFGF_NONE);
    install_validation_callbacks(all_opts);

    int ret = cfg_parse(all_opts, cfg_file.c_str());
    if (ret == CFG_FILE_ERROR) {
        assert(false && "Failed to open configuration file!");
    } else if (ret == CFG_PARSE_ERROR) {
        fprintf(stderr,
                "An error occurred when reading the configuration file!\n");
        exit(-1);
    }

    cfg_t* network_opts = cfg_getsec(all_opts, "network");
    int num_layers = cfg_size(network_opts, "layer");
    Network* network = new Network(cfg_getstr(network_opts, "name"));
    readInputDataConfig(network, workspace, network_opts);
    for (int i = 0; i < num_layers; i++) {
        cfg_t* opConfig = cfg_getnsec(network_opts, "layer", i);
        int numInputs = cfg_size(opConfig, "inputs");
        std::vector<Operator*> inputs;
        for (int i = 0; i < numInputs; i++) {
            std::string inputName = cfg_getnstr(opConfig, "inputs", i);
            inputs.push_back(network->getLayerLastOperator(inputName));
        }
        std::string layerName = cfg_title(opConfig);
        std::string type = cfg_getstr(opConfig, "type");
        createAndAddOperator(layerName, type, network, workspace, opConfig, inputs);
        std::string actfunc = cfg_getstr(opConfig, "activation");
        std::string actName = layerName + "_" + actfunc;
        createAndAddOperator(actName, actfunc, network, workspace, opConfig);
        Operator* lastOp = network->getLastOperator();
        network->addLayerLastOperator(layerName, lastOp);
    }

    // Allocate storage for all of the tensors. Assume float32 data type (this
    // can be customized if necessary).
    for (auto iter = network->begin(); iter != network->end(); ++iter) {
        Operator* op = iter->second;
        for (auto input : op->getInputs()) {
            Tensor<GlobalBackend>* tensor =
                    dynamic_cast<Tensor<GlobalBackend>*>(input);
            tensor->allocateStorage<float>();
        }
        for (auto output : op->getOutputs()) {
            Tensor<GlobalBackend>* tensor =
                    dynamic_cast<Tensor<GlobalBackend>*>(output);
            tensor->allocateStorage<float>();
        }
    }
#if 0
    // network->addDataLayoutTransformations<GlobalBackend>(workspace);

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
#endif
    network->printSummary();
    cfg_free(all_opts);
    return network;
}
