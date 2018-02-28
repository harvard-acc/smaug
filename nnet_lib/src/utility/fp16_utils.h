#ifndef _FP16_UTILS_H_
#define _FP16_UTILS_H_

#include "fp16.h"

#ifdef __F16C__

// Use the hardware support directly.
#include <x86intrin.h>

#define _CVT_PS_PH_256(p8_fp32_data, rounding_mode)                            \
    _mm256_cvtps_ph(p8_fp32_data, rounding_mode)

#define _CVT_PH_PS_256(p8_fp32_data) _mm256_cvtph_ps(p8_fp32_data)

#else

#ifdef TRACE_MODE
#warning "No F16C: LLVM-Tracer cannot emit IR FP convert instructions!"
#endif

// Call the FP16 library.
#define _CVT_PS_PH_256(fp32x8_data, rounding_mode)                     \
    {                                                                  \
        fp16_ieee_from_fp32_value(fp32x8_data[0]),                     \
        fp16_ieee_from_fp32_value(fp32x8_data[1]),                     \
        fp16_ieee_from_fp32_value(fp32x8_data[2]),                     \
        fp16_ieee_from_fp32_value(fp32x8_data[3]),                     \
        fp16_ieee_from_fp32_value(fp32x8_data[4]),                     \
        fp16_ieee_from_fp32_value(fp32x8_data[5]),                     \
        fp16_ieee_from_fp32_value(fp32x8_data[6]),                     \
        fp16_ieee_from_fp32_value(fp32x8_data[7])                      \
    }

#define _CVT_PH_PS_256(fp16x8_data)                                    \
    {                                                                  \
        fp16_ieee_to_fp32_value(fp16x8_data[0]),                       \
        fp16_ieee_to_fp32_value(fp16x8_data[1]),                       \
        fp16_ieee_to_fp32_value(fp16x8_data[2]),                       \
        fp16_ieee_to_fp32_value(fp16x8_data[3]),                       \
        fp16_ieee_to_fp32_value(fp16x8_data[4]),                       \
        fp16_ieee_to_fp32_value(fp16x8_data[5]),                       \
        fp16_ieee_to_fp32_value(fp16x8_data[6]),                       \
        fp16_ieee_to_fp32_value(fp16x8_data[7])                        \
    }

#endif  // __F16C__

// Although ISA support exists for this instruction in F16C, LLVM did not add
// support for this intrinsic until 3.8.
#define _CVT_SS_SH(val, rounding_mode) fp16_ieee_from_fp32_value(val)

#endif
