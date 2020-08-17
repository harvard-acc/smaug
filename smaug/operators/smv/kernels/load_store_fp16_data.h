/**
 * \file load_store_fp16_data.h
 * \brief Aladdin kernels to load/store FP16 data to/from host memory.
 */

#ifndef _OPERATORS_SMV_KERNELS_LOAD_STORE_FP16_DATA_H_
#define _OPERATORS_SMV_KERNELS_LOAD_STORE_FP16_DATA_H_

#include "smaug/utility/fp16_utils.h"
#include "smaug/operators/common.h"

#ifdef __cplusplus
extern "C" {
#endif

/** \ingroup AladdinKernels
 *
 * Loads half-precision fp data from the host and locally on the accelerator
 * converts it into single-precision data.
 *
 * The transfer operation is pipelined so it can be overlapped with the
 * conversion operation. Each transfer is at most one page in size (4KB), which
 * is converted into 8KB of data. The conversion is done in-place, so no
 * additional SRAM is required to buffer the FP16 data.
 *
 * @param local_data Single-precision accelerator-local scratchpad.
 * @param remote_data Half-precision host memory address.
 * @param num_elems Number of elements to copy.
 * @param local_offset Offset into local array to start copying data to in
 *        elements.
 * @param remote_offset Offset into remote memory to start copying data from in
 *        elements.
 */
void host_load_fp16(float* local_data,
                    float16* remote_data,
                    int num_elems,
                    int local_offset,
                    int remote_offset);

/** \ingroup AladdinKernels
 *
 * Converts single-precision fp data from the accelerator into half-precision
 * data and copy it to the host.
 *
 * The transfer operation is pipelined so it can be overlapped with the
 * conversion operation. Each transfer is at most one page in size (4KB), which
 * is converted from 8KB of data. The conversion is done in-place, so no
 * additional SRAM is required to buffer the FP16 data.
 *
 * @param local_data Single-precision accelerator-local scratchpad.
 * @param remote_data Half-precision host memory address.
 * @param num_elems Number of elements to copy.
 * @param local_offset Offset into local array to start copying data to in
 *        elements.
 * @param remote_offset Offset into remote memory to start copying data from in
 *        elements.
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
