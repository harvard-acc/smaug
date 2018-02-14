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

const char* get_host_inputs_var_name(io_req_t req) {
    return req == IO_DMA
                   ? "dma_activations"
                   : req == IO_ACP ? "acp_activations" : "cache_activations";
}

const char* get_host_weights_var_name(io_req_t req) {
    return req == IO_DMA ? "dma_weights"
                         : req == IO_ACP ? "acp_weights" : "cache_weights";
}

const char* get_host_results_var_name(io_req_t req) {
    return req == IO_DMA ? "dma_results"
                         : req == IO_ACP ? "acp_results" : "cache_results";
}
