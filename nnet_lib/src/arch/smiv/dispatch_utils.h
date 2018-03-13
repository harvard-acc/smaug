#ifndef _ARCH_DISPATCH_UTILS_H_
#define _ARCH_DISPATCH_UTILS_H_

#include "core/nnet_fwd_defs.h"
#include "core/smiv/params.h"

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
// Helper functions to do a bulk load/store over ACP.
// They should be called with host_src or host_dest as some variable prefixed
// with acp.
//
// coherentLoad32/coherentStore32 move data at 32-byte granularity. For better
// performance, use coherentLoad64/coherentStore64, which moves data at 64-byte
// granularity, the same size as the cache lines.
//
// *_offset_32b is a 32-byte-aligned offset into either the source and
// destination from which to start copying data, used for the 32-byte aligned
// variants.
//
// *_offset_64b is a 64-byte-aligned offset into either the source and
// destination from which to start copying data, used for the 64-byte aligned
// variants.
//
// coherentLoad16/coherentStore16 are used to load or store a single 16-byte
// value. They handle the requirement that all accesses to local storage be at
// 32-byte granularity and size.

ALWAYS_INLINE
static inline void coherentLoad16(float* local_dest,
                                  float* host_src,
                                  int local_offset_16b,
                                  int host_offset_16b) {
    VEC_ARRAY_1D(v8fp_t, _local_dest, local_dest);
    VEC_ARRAY_1D(v4fp_t, _host_src, host_src);
    v4fp_t host_data = _host_src[host_offset_16b];
    int local_dest_32b = local_offset_16b / 2;
    int vec_offset = (local_offset_16b % 2) * 4;
    // Perform a read-modify-write operation.
    v8fp_t local_data = _local_dest[local_offset_16b / 2];
    local_data[vec_offset] = host_data[0];
    local_data[vec_offset + 1] = host_data[1];
    local_data[vec_offset + 2] = host_data[2];
    local_data[vec_offset + 3] = host_data[3];
    _local_dest[local_dest_32b] = local_data;
}

ALWAYS_INLINE
static inline void coherentStore16(float* host_dest,
                                   float* local_src,
                                   int host_offset_16b,
                                   int local_offset_16b) {
    VEC_ARRAY_1D(v8fp_t, _local_src, local_src);
    VEC_ARRAY_1D(v4fp_t, _host_dest, host_dest);
    int local_src_32b = local_offset_16b / 2;
    int vec_offset = (local_offset_16b % 2) * 4;
    // Read 32-bytes, then extract the 16 bytes we want.
    v8fp_t local_data = _local_src[local_src_32b];
    v4fp_t host_data;
    host_data[0] = local_data[vec_offset + 0];
    host_data[1] = local_data[vec_offset + 1];
    host_data[2] = local_data[vec_offset + 2];
    host_data[3] = local_data[vec_offset + 3];
    _host_dest[host_offset_16b] = host_data;
}

ALWAYS_INLINE
static inline void coherentLoad32(float* local_dest,
                                  float* host_src,
                                  int transfer_size,
                                  int local_offset_32b,
                                  int host_offset_32b) {
    VEC_ARRAY_1D(v8fp_t, _local_dest, local_dest);
    VEC_ARRAY_1D(v8fp_t, _host_src, host_src);
    int elems = transfer_size / sizeof(float) / VECTOR_SIZE;
    loop:
    for (int i = 0; i < elems; i++) {
        _local_dest[i + local_offset_32b] = _host_src[i + host_offset_32b];
    }
}

ALWAYS_INLINE
static inline void coherentStore32(float* host_dest,
                                   float* local_src,
                                   int transfer_size,
                                   int host_offset_32b,
                                   int local_offset_32b) {
    VEC_ARRAY_1D(v8fp_t, _host_dest, host_dest);
    VEC_ARRAY_1D(v8fp_t, _local_src, local_src);
    int elems = transfer_size / sizeof(float) / VECTOR_SIZE;
    loop:
    for (int i = 0; i < elems; i++) {
        _host_dest[i + host_offset_32b] = _local_src[i + local_offset_32b];
    }
}

ALWAYS_INLINE
static inline void coherentLoad64(float* local_dest,
                                  float* host_src,
                                  int transfer_size,
                                  int local_offset_64b,
                                  int host_offset_64b) {
    ASSERT(transfer_size % VECTOR_SIZE == 0 &&
           "Transfer size must be a multiple of VECTOR_SIZE!");
    VEC_ARRAY_1D(v16fp_t, _local_dest_64, local_dest);
    VEC_ARRAY_1D(v16fp_t, _host_src_64, host_src);

    int elems_64 = transfer_size / CACHELINE_SIZE;
    bool has_remaining = transfer_size % CACHELINE_SIZE != 0;
    // First copy as much of the data via full cacheline loads.
    loop:
    for (int i = 0; i < elems_64; i++) {
        _local_dest_64[i + local_offset_64b] =
                _host_src_64[i + host_offset_64b];
    }
    // Finish off the remainder with a RMW operation. This is slow since we
    // need to first read the host data, update the elements we want, and then
    // send it back.
    if (has_remaining) {
        v16fp_t host_cacheline = _host_src_64[elems_64 + host_offset_64b];
        v16fp_t local_cacheline = _local_dest_64[elems_64 + local_offset_64b];
        int remaining_qwords = transfer_size % CACHELINE_SIZE;
        rmw:
        for (int i = 0; i < remaining_qwords; i++) {
            local_cacheline[i] = host_cacheline[i];
        }
        _local_dest_64[elems_64] = local_cacheline;
    }
}

ALWAYS_INLINE
static inline void coherentStore64(float* host_dest,
                                   float* local_src,
                                   int transfer_size,
                                   int host_offset_64b,
                                   int local_offset_64b) {
    ASSERT(transfer_size % VECTOR_SIZE == 0 &&
           "Transfer size must be a multiple of VECTOR_SIZE!");
    VEC_ARRAY_1D(v16fp_t, _host_dest_64, host_dest);
    VEC_ARRAY_1D(v16fp_t, _local_src_64, local_src);
    int elems_64 = transfer_size / CACHELINE_SIZE;
    int remainder = transfer_size % CACHELINE_SIZE;
    // First copy as much of the data via full cacheline stores.
    loop:
    for (int i = 0; i < elems_64; i++) {
        _host_dest_64[i + host_offset_64b] =
                _local_src_64[i + local_offset_64b];
    }
    // Finish off the remainder with a RMW operation. This is slow since we
    // need to first read the host data, update the elements we want, and then
    // send it back.
    if (remainder > 0) {
        v16fp_t host_cacheline = _host_dest_64[elems_64 + host_offset_64b];
        v16fp_t local_cacheline = _local_src_64[elems_64 + local_offset_64b];
        int remaining_qwords = transfer_size % CACHELINE_SIZE;
        rmw:
        for (int i = 0; i < remaining_qwords; i++) {
            host_cacheline[i] = local_cacheline[i];
        }
        _host_dest_64[elems_64] = host_cacheline;
    }
}

ALWAYS_INLINE
static inline void coherentStore(float* host_dest,
                                 float* local_src,
                                 int transfer_size,
                                 int host_offset_4b,
                                 int local_offset_4b) {
    // Handle unaligned store at the beginning.
    ASSERT(transfer_size % 16 == 0 &&
           "Transfer size must be a multiple of 16 bytes!");
    if (transfer_size % 32 == 16) {
        coherentStore16(
                host_dest, local_src, host_offset_4b / 4, local_offset_4b / 4);
        transfer_size -= 16;
        host_offset_4b += 4;
        local_offset_4b += 4;
    }
    coherentStore64(host_dest,
                    local_src,
                    transfer_size,
                    host_offset_4b / 16,
                    local_offset_4b / 16);
}

//=--------------------------------------------------------------------------=//
// Returns the canonical function arg name for this access mechanism.

const char* get_host_inputs_var_name(io_req_t req);
const char* get_host_weights_var_name(io_req_t req);
const char* get_host_results_var_name(io_req_t req);

#endif
