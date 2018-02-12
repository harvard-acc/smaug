#include "confuse.h"

cfg_opt_t convolution_param_cfg[] = {
    CFG_STR("type", "STANDARD", CFGF_NONE),
    CFG_INT("num_output", -1, CFGF_NODEFAULT),
    CFG_INT("pad", -1, CFGF_NODEFAULT),
    CFG_INT("kernel_size", -1, CFGF_NODEFAULT),
    CFG_INT("stride", 1, CFGF_NODEFAULT), CFG_END()
};

cfg_opt_t inner_product_param_cfg[] = {
    CFG_INT("num_output", -1, CFGF_NODEFAULT), CFG_END()
};

cfg_opt_t pooling_param_cfg[] = { CFG_STR("pool", "", CFGF_NODEFAULT),
                                  CFG_INT("size", -1, CFGF_NODEFAULT),
                                  CFG_INT("stride", -1, CFGF_NODEFAULT),
                                  CFG_END() };

cfg_opt_t layer_cfg[] = {
    CFG_STR("type", "", CFGF_NODEFAULT),
    CFG_STR("activation", "NONE", CFGF_NONE),
    CFG_SEC("convolution_param", convolution_param_cfg, CFGF_NODEFAULT),
    CFG_SEC("inner_product_param", inner_product_param_cfg, CFGF_NODEFAULT),
    CFG_SEC("pooling_param", pooling_param_cfg, CFGF_NODEFAULT),
    CFG_END()
};

cfg_opt_t network_cfg[] = {
    CFG_STR("name", "network", CFGF_NONE),
    CFG_INT("input_rows", -1, CFGF_NODEFAULT),
    CFG_INT("input_cols", -1, CFGF_NODEFAULT),
    CFG_INT("input_height", -1, CFGF_NODEFAULT),
    CFG_SEC("layer", layer_cfg, CFGF_MULTI | CFGF_TITLE | CFGF_NO_TITLE_DUPES),
    CFG_END()
};

cfg_opt_t device_cfg[] = {
  CFG_STR("cpu_default_offload", "DMA", CFGF_NONE),
  CFG_STR("cpu_pooling_offload", "DMA", CFGF_NONE),
  CFG_STR("cpu_activation_func_offload", "DMA", CFGF_NONE),
  CFG_BOOL("use_hw_activation_func", cfg_true, CFGF_NONE),
  CFG_BOOL("use_hw_batch_norm", cfg_false, CFGF_NONE),
  CFG_BOOL("use_hw_pooling", cfg_false, CFGF_NONE),
  CFG_BOOL("use_pipelined_dma", cfg_false, CFGF_NONE),
  CFG_END()
};

cfg_opt_t sampling_cfg[] = {
  CFG_INT("standard_conv_num_filters", 0, CFGF_NONE),
  CFG_INT("fc_num_neurons", 0, CFGF_NONE),
  CFG_INT("smv_conv_inner_iters", 0, CFGF_NONE),
  CFG_END()
};

cfg_opt_t top_level_cfg[] = {
    CFG_SEC("network", network_cfg, CFGF_NODEFAULT),
    CFG_SEC("device", device_cfg, CFGF_NODEFAULT),
    CFG_SEC("sampling_param", sampling_cfg, CFGF_NODEFAULT),
    CFG_END(),
};
