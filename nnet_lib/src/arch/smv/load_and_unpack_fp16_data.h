#include "arch/smiv/dispatch_utils.h"
#include "core/smiv/params.h"
#include "utility/fp16_utils.h"

/* Use DMA or ACP to load FP16 data and unpack it into a local scratchpad.
 *
 * The DMA operation is pipelined so it can be overlapped with the unpacking
 * operation. Each DMA transfer is at most one page in size (4KB), which is
 * unpacked into 8KB of data. ACP also copies data at 4KB granualarity. The
 * unpacking is done in-place, so no additional SRAM is required to buffer the
 * packed data.
 *
 * Args:
 *   local_data: The local buffer in which the results should be stored.
 *   remote_data: The host buffer to load from.
 *   num_elems: The number of elements to load. For the initial DMA load, each
 *     element is assumed to be 16 bits in size.
 *   local_offset: Start from this offset in the local buffer. This offset is
 *     a 4-byte element-wise offset.
 *   remote_offset: Start from this offset in the remote buffer.
 *   use_dma: Use DMA if true, ACP if false.
 */
ALWAYS_INLINE
static inline void load_and_unpack_fp16(float* local_data,
                                        packed_fp16* remote_data,
                                        int num_elems,
                                        int local_offset,
                                        int remote_offset,
                                        bool use_dma) {
    VEC_ARRAY_1D(v16ph_t, _local_data_hp, local_data);
    VEC_ARRAY_1D(v8fp_t, _local_data_sp, local_data);
    const int page_size = (1 << LOG_PAGE_SIZE);
    const int max_transfer_size = page_size;
    const int total_bytes = num_elems * sizeof(short);
    int num_xfers = FRAC_CEIL(total_bytes, max_transfer_size);
    int num_bytes_remaining = total_bytes;
    dma_unpack:
    for (int i = 0; i < num_xfers; i++) {
        int transfer_size = min2(num_bytes_remaining, max_transfer_size);
        int curr_local_offset = (i * page_size * 2) / sizeof(float);
        int curr_remote_offset = curr_local_offset / 2;
        if (use_dma) {
            dmaLoad(local_data + local_offset + curr_local_offset,
                    remote_data + remote_offset + curr_remote_offset,
                    transfer_size);
        } else {
            coherentLoad64(
                    local_data,
                    (float*)remote_data,
                    transfer_size,
                    (local_offset + curr_local_offset) / VECTOR_SIZE / 2,
                    (remote_offset + curr_remote_offset) / VECTOR_SIZE / 2);
        }

        // This loads N bytes of packed data into local_data. We now expand
        // N bytes of half precision to 2*N bytes of single precision, in
        // place, 32 bytes at a time.  In order to do this without overwriting
        // the data we're trying to unpack, we need to start from the back.
        int num_vectors = FRAC_CEIL(transfer_size, VECTOR_SIZE * sizeof(float));
        int page_offset_vec = (local_offset + curr_local_offset) / VECTOR_SIZE;
        vector_unpack:
        for (int v = num_vectors - 1; v >= 0; v--) {
            v16ph_t packed_data = _local_data_hp[page_offset_vec + v];
            v8ph_t packed_lo_8 = { packed_data[0], packed_data[1],
                                   packed_data[2], packed_data[3],
                                   packed_data[4], packed_data[5],
                                   packed_data[6], packed_data[7] };
            v8ph_t packed_hi_8 = { packed_data[8],  packed_data[9],
                                   packed_data[10], packed_data[11],
                                   packed_data[12], packed_data[13],
                                   packed_data[14], packed_data[15] };
            v8fp_t unpacked_lo_8 = _CVT_PH_PS_256(packed_lo_8);
            v8fp_t unpacked_hi_8 = _CVT_PH_PS_256(packed_hi_8);
            _local_data_sp[page_offset_vec + 2 * v] =  unpacked_lo_8;
            _local_data_sp[page_offset_vec + 2 * v + 1] = unpacked_hi_8;
        }
        num_bytes_remaining -= transfer_size;
    }
}

/* Use DMA or ACP to pack FP32 data and store it to a remote buffer.
 *
 * The DMA operation is pipelined so it can be overlapped with the packing
 * operation. Each DMA transfer is at most one page in size (4KB), which is
 * packed from 8KB of data. ACP also copies data at 4KB granualarity. The
 * packing is done in-place, so no additional SRAM is required to buffer the
 * packed data.
 *
 * Args:
 *   local_data: The local buffer holding the FP32 data to store.
 *   remote_data: The host buffer to store to.
 *   num_elems: The number of elements to store.
 *   local_offset: Start from this offset in the local buffer. This offset is
 *     a 4-byte element-wise offset.
 *   remote_offset: Start from this offset in the remote buffer.
 *   use_dma: Use DMA if true, ACP if false.
 */
ALWAYS_INLINE
static inline void pack_and_store_fp16(float* local_data,
                                       packed_fp16* remote_data,
                                       int num_elems,
                                       int local_offset,
                                       int remote_offset,
                                       bool use_dma) {
    VEC_ARRAY_1D(v16ph_t, _local_data_hp, local_data);
    VEC_ARRAY_1D(v8fp_t, _local_data_sp, local_data);
    const int page_size = (1 << LOG_PAGE_SIZE);
    const int max_transfer_size = page_size;
    const int total_bytes = num_elems * sizeof(float16);
    int num_xfers = FRAC_CEIL(total_bytes, max_transfer_size);
    int num_bytes_remaining = total_bytes;
    // packed_fp16 is a 4-byte type, but float16 is 2 bytes, so in order to
    // index into it on a 2-byte granularity, we have to halve the offset.
    remote_offset /= 2;
    dma_pack:
    for (int i = 0; i < num_xfers; i++) {
        int transfer_size = min2(num_bytes_remaining, max_transfer_size);
        // The effective transfer size is the size in terms of FP32.
        int eff_transfer_size = transfer_size * 2;
        int curr_local_offset = (i * 2 * page_size) / sizeof(float);
        int curr_remote_offset = curr_local_offset / 2;

        int num_vectors =
                FRAC_CEIL(eff_transfer_size, VECTOR_SIZE * sizeof(float));
        int page_offset_vec = (local_offset + curr_local_offset) / VECTOR_SIZE;
        vector_pack:
        for (int v = 0; v < num_vectors; v += 2){
            v8fp_t unpacked_lo_8 = _local_data_sp[page_offset_vec + v];
            v8fp_t unpacked_hi_8 = _local_data_sp[page_offset_vec + v + 1];
            v8ph_t packed_lo_8 = _CVT_PS_PH_256(unpacked_lo_8, 0);
            v8ph_t packed_hi_8 = _CVT_PS_PH_256(unpacked_hi_8, 0);

            v16ph_t packed_data = {
                packed_lo_8[0], packed_lo_8[1], packed_lo_8[2], packed_lo_8[3],
                packed_lo_8[4], packed_lo_8[5], packed_lo_8[6], packed_lo_8[7],
                packed_hi_8[0], packed_hi_8[1], packed_hi_8[2], packed_hi_8[3],
                packed_hi_8[4], packed_hi_8[5], packed_hi_8[6], packed_hi_8[7]
            };
            _local_data_hp[page_offset_vec + v / 2] = packed_data;
        }
        if (use_dma) {
            dmaStore(remote_data + remote_offset + curr_remote_offset,
                     local_data + local_offset + curr_local_offset,
                     transfer_size);
        } else {
            coherentStore64(
                    (float*)remote_data,
                    local_data,
                    transfer_size,
                    (remote_offset + curr_remote_offset) / VECTOR_SIZE / 2,
                    (local_offset + curr_local_offset) / VECTOR_SIZE / 2);
        }

        num_bytes_remaining -= transfer_size;
    }
}

ALWAYS_INLINE
static inline void dma_load_and_unpack_fp16(float* local_data,
                                            packed_fp16* remote_data,
                                            int num_elems,
                                            int local_offset,
                                            int remote_offset) {
    load_and_unpack_fp16(local_data,
                         remote_data,
                         num_elems,
                         local_offset,
                         remote_offset,
                         true);
}

ALWAYS_INLINE
static inline void acp_load_and_unpack_fp16(float* local_data,
                                            packed_fp16* remote_data,
                                            int num_elems,
                                            int local_offset,
                                            int remote_offset) {
    load_and_unpack_fp16(local_data,
                         remote_data,
                         num_elems,
                         local_offset,
                         remote_offset,
                         false);
}

ALWAYS_INLINE
static inline void dma_pack_and_store_fp16(packed_fp16* remote_data,
                                           float* local_data,
                                           int num_elems,
                                           int remote_offset,
                                           int local_offset) {
    pack_and_store_fp16(local_data,
                        remote_data,
                        num_elems,
                        local_offset,
                        remote_offset,
                        true);
}

ALWAYS_INLINE
static inline void acp_pack_and_store_fp16(packed_fp16* remote_data,
                                           float* local_data,
                                           int num_elems,
                                           int remote_offset,
                                           int local_offset) {
    pack_and_store_fp16(local_data,
                        remote_data,
                        num_elems,
                        local_offset,
                        remote_offset,
                        false);
}
