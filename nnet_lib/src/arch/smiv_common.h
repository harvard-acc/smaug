#ifndef _ARCH_SMIV_COMMON_H_
#define _ARCH_SMIV_COMMON_H_

#if ARCHITECTURE == SMIV

#include "nnet_fwd.h"
#include "utility/profiling.h"

// Convenience macro for profiling a kernel invocation.
#define INVOKE_KERNEL_PROF(req_code, lnum, kernel_ptr, args...)                \
    do {                                                                       \
        begin_profiling(STRING(kernel_ptr), lnum);                             \
        INVOKE_KERNEL(req_code, kernel_ptr, args);                             \
        end_profiling();                                                       \
    } while (0)

// Each SMIV block has two scratchpads of 64KB each, but the real accelerator
// operates on 16-bit data, whereas we are using 32-bit data. To make sure we
// can fit the same size inputs, we double the per-scratchpad size.
#define SPAD_SIZE (131072)

// The UMEM on the NIC is 3 blocks of 1MB each.
#define UMEM_SIZE (3*1048576)

// A struct that defines how work will be divided across iterations.
typedef struct _work_cfg_t {
    // An array of dim_t objects. Specify the rows, cols, channels, and padding
    // for each iteration.
    dims_t* iteration;
    // Number of iterations that are required.
    unsigned num_iterations;
} work_cfg_t;
typedef work_cfg_t fc_cfg_t;
typedef work_cfg_t conv_cfg_t;
typedef work_cfg_t pool_cfg_t;

void init_work_cfg(work_cfg_t* cfg, unsigned num_iterations);
void free_work_cfg(work_cfg_t* cfg);
void print_work_cfg(work_cfg_t* cfg);

// Accelerator id codes.
extern unsigned kConvolutionHw;
extern unsigned kInnerProductHw;
extern unsigned kReductionHw;
extern unsigned kBatchNormHw;
extern unsigned kPoolingHw;

// SMIV SRAM structures.
extern float* g_umem;
extern float* g_spad0;
extern float* g_spad1;

// Returns whether the hardware can execute this activation function on this
// layer type or not.
bool is_supported_activation_func(layer_type ltype, activation_type func);

result_buf smiv_activation_function(float* activations,
                                    layer_t* layer,
                                    float* results,
                                    device_t* device);

// These functions handle the task of breaking up a layer's input and weights
// into blocks, individually running them on the accelerator, and aggregating
// the results.
void standard_convolution_layer_impl(float* host_activations,
                                     float* host_weights,
                                     layer_t* layers,
                                     int lnum,
                                     float* host_result,
                                     device_t* device,
                                     sampling_param_t* sampling_param);
bool inner_product_needs_work_division(layer_t* layer);
void inner_product_layer_impl(float* host_activations,
                              float* host_weights,
                              layer_t* layers,
                              int lnum,
                              float* host_result,
                              device_t* device);
void depthwise_convolution_layer_impl(float* host_activations,
                                      float* host_weights,
                                      layer_t* layers,
                                      int lnum,
                                      float* host_result,
                                      device_t* device);

void pooling_layer_impl(float* inputs, layer_t* curr_layer, float* results);

#endif  // ARCHITECTURE == SMIV
#endif
