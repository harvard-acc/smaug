#include "arch/smiv/dispatch_utils.h"
#include "core/nnet_fwd_defs.h"
#include "utility/utility.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

ALWAYS_INLINE
access_mechanism io_to_access_mechanism(io_req_t req) {
    switch (req) {
        case IO_NONE:
        case IO_DMA:
            return _DmaOrLocal;
        case IO_ACP:
            return _ACP;
        case IO_CACHE:
            return _Cache;
        default:
            assert(false && "Unknown io_req_t value!");
            return _UnknownMechanism;
    }
}

ALWAYS_INLINE
access_config layer_to_access_config(layer_t* curr_layer) {
    access_config access_config;
    access_config.inputs = io_to_access_mechanism(curr_layer->input_req);
    access_config.weights = io_to_access_mechanism(curr_layer->weights_req);
    access_config.outputs = io_to_access_mechanism(curr_layer->output_req);
    return access_config;
}

ALWAYS_INLINE
void dma_wrapper(float* host,
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
void dma_load_wrapper(float* local_dest,
                      float* host_src,
                      size_t transfer_size,
                      bool use_pipelined_dma) {
    dma_wrapper(host_src, local_dest, transfer_size, true, use_pipelined_dma);
}

ALWAYS_INLINE
void dma_store_wrapper(float* host_dest,
                       float* local_src,
                       size_t transfer_size,
                       bool use_pipelined_dma) {
    dma_wrapper(host_dest, local_src, transfer_size, false, use_pipelined_dma);
}
