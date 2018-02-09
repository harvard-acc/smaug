#ifndef _ARCH_DISPATCH_UTILS_H_
#define _ARCH_DISPATCH_UTILS_H_

#include "core/nnet_fwd_defs.h"

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

//=--------------------------------------------------------------------------=//
// Wrapper functions to handle deciding between whether or not to use pipelined
// DMA.

void dma_wrapper(float* host,
                 float* local,
                 size_t transfer_size,
                 bool is_load,
                 bool use_pipelined_dma);

void dma_load_wrapper(float* local_dest,
                      float* host_src,
                      size_t transfer_size,
                      bool use_pipelined_dma);

void dma_store_wrapper(float* host_dest,
                       float* local_src,
                       size_t transfer_size,
                       bool use_pipelined_dma);

#endif
