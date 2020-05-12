#ifndef _OPERATORS_SMV_KERNELS_LOAD_STORE_FP16_DATA_H_
#define _OPERATORS_SMV_KERNELS_LOAD_STORE_FP16_DATA_H_

#include "smaug/utility/fp16_utils.h"
#include "smaug/operators/common.h"

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
void host_load_fp16(float* local_data,
                    float16* remote_data,
                    int num_elems,
                    int local_offset,
                    int remote_offset);

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
void host_store_fp16(float* local_data,
                     float16* remote_data,
                     int num_elems,
                     int local_offset,
                     int remote_offset);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
