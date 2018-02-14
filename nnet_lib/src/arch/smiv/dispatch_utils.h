#ifndef _ARCH_DISPATCH_UTILS_H_
#define _ARCH_DISPATCH_UTILS_H_

#include "core/nnet_fwd_defs.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

// A conditional statement that checks if an access mechanism configuration
// matches some combination. This can be used directly in an if statement.
#define DISPATCH_2(config, _inputs, _outputs)                                  \
    (((config)->inputs == (_inputs)) && ((config)->outputs == (_outputs)))

#define DISPATCH_3(config, _inputs, _weights, _outputs)                        \
    (((config)->inputs == (_inputs)) && ((config)->weights == (_weights)) &&   \
     ((config)->outputs == (_outputs)))

//=--------------------------------------------------------------------------=//
// These are INTERNAL data structures and functions that should only be used by
// the dispatcher! All other functions should directly use layer_t.*_req
// instead.

typedef enum _access_mechanism {
    _DmaOrLocal = IO_NONE,
    _ACP = IO_ACP,
    _Cache = IO_CACHE,
    _UnknownMechanism
} access_mechanism;

typedef struct _access_config {
    access_mechanism inputs;
    access_mechanism weights;
    access_mechanism outputs;
} access_config;

access_mechanism io_to_access_mechanism(io_req_t req);
access_config layer_to_access_config(layer_t* curr_layer);

// This function is used for pipelining DMA requst.
ALWAYS_INLINE
static inline void divide_and_send_dma_req(float* host_base,
                                           float* local_base,
                                           int size,
                                           int log2_dma_chunk_size,
                                           bool isLoad) {
    int dma_chunk_size = 1 << log2_dma_chunk_size;
    int num_dma_reqs = (size + dma_chunk_size - 1) >> log2_dma_chunk_size;
    int dma_req_size;
    int last_req_size = size - (num_dma_reqs - 1) * dma_chunk_size;

dma_division:
    for (int i = 0; i < num_dma_reqs; i++) {
        dma_req_size = (i == num_dma_reqs - 1) ? last_req_size : dma_chunk_size;
        if (isLoad) {
            dmaLoad(local_base + i * dma_chunk_size / sizeof(float),
                    host_base + i * dma_chunk_size / sizeof(float),
                    dma_req_size);
        } else {
            dmaStore(host_base + i * dma_chunk_size / sizeof(float),
                     local_base + i * dma_chunk_size / sizeof(float),
                     dma_req_size);
        }
    }
}

//=--------------------------------------------------------------------------=//
// Wrapper functions to handle deciding between whether or not to use pipelined
// DMA.

ALWAYS_INLINE
static inline void dma_wrapper(float* host,
                 float* local,
                 size_t transfer_size,
                 bool is_load,
                 bool use_pipelined_dma) {
    if (use_pipelined_dma) {
        divide_and_send_dma_req(
                host, local, transfer_size, LOG_PAGE_SIZE, is_load);
    } else {
        if (is_load)
            dmaLoad(local, host, transfer_size);
       else
            dmaStore(host, local, transfer_size);
    }
}

ALWAYS_INLINE
static inline void dma_load_wrapper(float* local_dest,
                      float* host_src,
                      size_t transfer_size,
                      bool use_pipelined_dma) {
    dma_wrapper(host_src, local_dest, transfer_size, true, use_pipelined_dma);
}

ALWAYS_INLINE
static inline void dma_store_wrapper(float* host_dest,
                       float* local_src,
                       size_t transfer_size,
                       bool use_pipelined_dma) {
    dma_wrapper(host_dest, local_src, transfer_size, false, use_pipelined_dma);
}

//=--------------------------------------------------------------------------=//
// Returns the canonical function arg name for this access mechanism.

const char* get_host_inputs_var_name(io_req_t req);
const char* get_host_weights_var_name(io_req_t req);
const char* get_host_results_var_name(io_req_t req);

#endif
