#ifndef _OPERATORS_SMV_KERNELS_LOAD_STORE_FP16_DATA_H_
#define _OPERATORS_SMV_KERNELS_LOAD_STORE_FP16_DATA_H_

#include "utility/fp16_utils.h"
#include "operators/common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Use DMA to load FP16 data into a local scratchpad and convert it into FP32.
 *
 * The DMA operation is pipelined so it can be overlapped with the conversion
 * operation. Each DMA transfer is at most one page in size (4KB), which is
 * converted into 8KB of data. The conversion is done in-place, so no additional
 * SRAM is required to buffer the FP16 data.
 *
 * Args:
 *   local_data: The local buffer in which the results should be stored.
 *   remote_data: The host buffer to load from.
 *   num_elems: The number of elements to load. For the initial DMA load, each
 *     element is assumed to be 16 bits in size.
 *   local_offset: Start from this offset in the local buffer. This offset is
 *     a 4-byte element-wise offset.
 *   remote_offset: Start from this offset in the remote buffer.
 */
ALWAYS_INLINE
static inline void dma_load_fp16(float* local_data,
                                 float16* remote_data,
                                 int num_elems,
                                 int local_offset,
                                 int remote_offset) {
    VEC_ARRAY_1D(v8ph_t, _local_data_hp, local_data);
    VEC_ARRAY_1D(v8fp_t, _local_data_sp, local_data);
    const int page_size = (1 << LOG_PAGE_SIZE);
    const int max_transfer_size = page_size;
    const int total_bytes =
            next_multiple(num_elems * sizeof(float16), CACHELINE_SIZE);
    int num_xfers = FRAC_CEIL(total_bytes, max_transfer_size);
    int num_bytes_remaining = total_bytes;
    dma_fp16_to_fp32:
    for (int i = 0; i < num_xfers; i++) {
        int transfer_size = min2(num_bytes_remaining, max_transfer_size);
        int curr_offset = (i * page_size * 2) / sizeof(float);
        dmaLoad(local_data + local_offset + curr_offset,
                remote_data + remote_offset + curr_offset,
                transfer_size);

        // This loads N bytes of FP16 data into local_data. We now expand
        // N bytes of half precision to 2*N bytes of single precision, in
        // place, 32 bytes at a time. In order to do this without overwriting
        // the data we're trying to unpack, we need to start from the back.
        int num_vectors =
                FRAC_CEIL(transfer_size * 2, VECTOR_SIZE * sizeof(float));
        int page_offset_vec = (local_offset + curr_offset) / VECTOR_SIZE;
        vector_fp16_to_fp32:
        for (int v = num_vectors - 1; v >= 0; v--) {
            v8ph_t fp16_data = _local_data_hp[page_offset_vec * 2 + v];
            v8fp_t fp32_data = _CVT_PH_PS_256(fp16_data);
            _local_data_sp[page_offset_vec + v] = fp32_data;
        }
        num_bytes_remaining -= transfer_size;
    }
}

/* Use DMA to convert FP32 data into FP16 and store it to a remote buffer.
 *
 * The DMA operation is pipelined so it can be overlapped with the conversion
 * operation. Each DMA transfer is at most one page in size (4KB), which is
 * converted from 8KB of data. The conversion is done in-place, so no additional
 * SRAM is required to buffer the FP16 data.
 *
 * Args:
 *   local_data: The local buffer holding the FP32 data to store.
 *   remote_data: The host buffer to store to.
 *   num_elems: The number of elements to store.
 *   local_offset: Start from this offset in the local buffer. This offset is
 *     a 4-byte element-wise offset.
 *   remote_offset: Start from this offset in the remote buffer.
 */
ALWAYS_INLINE
static inline void dma_store_fp16(float* local_data,
                                  float16* remote_data,
                                  int num_elems,
                                  int local_offset,
                                  int remote_offset) {
    VEC_ARRAY_1D(v8ph_t, _local_data_hp, local_data);
    VEC_ARRAY_1D(v8fp_t, _local_data_sp, local_data);
    const int page_size = (1 << LOG_PAGE_SIZE);
    const int max_transfer_size = page_size;
    const int total_bytes = next_multiple(num_elems * sizeof(float16), 16);
    int num_xfers = FRAC_CEIL(total_bytes, max_transfer_size);
    int num_bytes_remaining = total_bytes;
    dma_fp32_to_fp16:
    for (int i = 0; i < num_xfers; i++) {
        int transfer_size = min2(num_bytes_remaining, max_transfer_size);
        // The effective transfer size is the size in terms of FP32.
        int eff_transfer_size = transfer_size * 2;
        int curr_offset = (i * 2 * page_size) / sizeof(float);

        int num_vectors =
                FRAC_CEIL(eff_transfer_size, VECTOR_SIZE * sizeof(float));
        int page_offset_vec = (local_offset + curr_offset) / VECTOR_SIZE;
        vector_fp32_to_fp16:
        for (int v = 0; v < num_vectors; v++){
            v8fp_t fp32_data = _local_data_sp[page_offset_vec + v];
            v8ph_t fp16_data = _CVT_PS_PH_256(fp32_data, 0);
            _local_data_hp[page_offset_vec * 2 + v] = fp16_data;
        }

        transfer_size = next_multiple(transfer_size, 32);
        dmaStore(remote_data + remote_offset + curr_offset,
                 local_data + local_offset + curr_offset,
                 transfer_size);

        num_bytes_remaining -= transfer_size;
    }
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
