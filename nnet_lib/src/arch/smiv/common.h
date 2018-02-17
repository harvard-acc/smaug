#ifndef _ARCH_SMIV_COMMON_H_
#define _ARCH_SMIV_COMMON_H_

#include "nnet_fwd.h"
#include "utility/profiling.h"

// Globally visible SMIV functions and types.
//
// **IMPORTANT**
//
// Any new addition to this file MUST have its name prefixed with smiv_ or
// 'smiv' placed in an otherwise logically consistent location.

// Each SMIV block has two scratchpads of 64KB each, but the real accelerator
// operates on 16-bit data, whereas we are using 32-bit data. To make sure we
// can fit the same size inputs, we double the per-scratchpad size.
#define SMIV_SPAD_SIZE (131072)

// The UMEM on the NIC is 3 blocks of 1MB each.
#define SMIV_UMEM_SIZE (3*1048576)

// These are GLOBAL arrays which cannot be referenced directly by a HW
// function. Instead, pass them to the top level functions as function
// arguments, and use a boolean flag to indicate which one contains the data
// needed.
typedef struct _smiv_global {
    float* umem;
    float* spad0;
    float* spad1;
    unsigned kConvolutionHw;
    unsigned kInnerProductHw;
    unsigned kReductionHw;
    unsigned kBatchNormHw;
    unsigned kPoolingHw;
} smiv_global;

extern smiv_global g_smiv;

// A struct that defines how work will be divided across iterations.
typedef struct _smiv_work_cfg_t {
    // An array of dim_t objects. Specify the rows, cols, channels, and padding
    // for each iteration.
    dims_t* iteration;
    // Number of iterations that are required.
    unsigned num_iterations;
} smiv_work_cfg_t;
typedef smiv_work_cfg_t fc_cfg_t;
typedef smiv_work_cfg_t conv_cfg_t;
typedef smiv_work_cfg_t pool_cfg_t;

void init_smiv_work_cfg(smiv_work_cfg_t* cfg, unsigned num_iterations);
void free_smiv_work_cfg(smiv_work_cfg_t* cfg);
void print_smiv_work_cfg(smiv_work_cfg_t* cfg);

// Returns whether the hardware can execute this activation function on this
// layer type or not.
bool smiv_is_supported_activation_func(layer_type ltype, activation_type func);
bool smiv_inner_product_needs_work_division(layer_t* curr_layer);
void smiv_inner_product_check_absolute_size_limits(layer_t* curr_layer);

result_buf smiv_activation_function(float* activations,
                                    layer_t* layer,
                                    float* results,
                                    device_t* device);

// These functions tile an input to fit on the accelerator's local memory.
void smiv_standard_convolution_layer_impl(float* host_activations,
                                          float* host_weights,
                                          layer_t* layers,
                                          int lnum,
                                          float* host_result,
                                          smiv_global* g_smiv,
                                          device_t* device,
                                          sampling_param_t* sampling_param);
void smiv_inner_product_layer_impl(float* host_activations,
                                   float* host_weights,
                                   layer_t* layers,
                                   int lnum,
                                   float* host_result,
                                   smiv_global* g_smiv,
                                   device_t* device);
void smiv_depthwise_convolution_layer_impl(float* host_activations,
                                           float* host_weights,
                                           layer_t* layers,
                                           int lnum,
                                           float* host_result,
                                           smiv_global* g_smiv,
                                           device_t* device);
void smiv_batch_norm_layer_impl(float* activations,
                                float* weights,
                                layer_t* layers,
                                int lnum,
                                float* result,
                                smiv_global* g_smiv,
                                device_t* device);
void smiv_pooling_layer_impl(float* inputs,
                             layer_t* curr_layer,
                             smiv_global* g_smiv,
                             float* results);
void smiv_decompress_packed_csr_impl(layer_t* layer,
                                     int weights_list_idx,
                                     int start_row,
                                     bool input_in_spad0,
                                     smiv_global* g_smiv,
                                     device_t* device);

#endif
