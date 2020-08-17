#ifndef _FP16_UTILS_H_
#define _FP16_UTILS_H_

#include <x86intrin.h>
#include "fp16.h"

//=------------- SW emulation of conversion instructions --------------=//


/**
 * Converts each element of an 8-wide vector of single-precision FP values to
 * half-precision FP, using the FP16 library.
 */
#define _SW_CVT_PS_PH_256(fp32x8_data, rounding_mode)                  \
    {                                                                  \
        fp16_ieee_from_fp32_value((fp32x8_data)[0]),                   \
        fp16_ieee_from_fp32_value((fp32x8_data)[1]),                   \
        fp16_ieee_from_fp32_value((fp32x8_data)[2]),                   \
        fp16_ieee_from_fp32_value((fp32x8_data)[3]),                   \
        fp16_ieee_from_fp32_value((fp32x8_data)[4]),                   \
        fp16_ieee_from_fp32_value((fp32x8_data)[5]),                   \
        fp16_ieee_from_fp32_value((fp32x8_data)[6]),                   \
        fp16_ieee_from_fp32_value((fp32x8_data)[7])                    \
    }

/**
 * Converts each element of an 8-wide vector of half-precision FP values to
 * single-precision FP, using the FP16 library.
 */
#define _SW_CVT_PH_PS_256(fp16x8_data)                                 \
    {                                                                  \
        fp16_ieee_to_fp32_value((fp16x8_data)[0]),                     \
        fp16_ieee_to_fp32_value((fp16x8_data)[1]),                     \
        fp16_ieee_to_fp32_value((fp16x8_data)[2]),                     \
        fp16_ieee_to_fp32_value((fp16x8_data)[3]),                     \
        fp16_ieee_to_fp32_value((fp16x8_data)[4]),                     \
        fp16_ieee_to_fp32_value((fp16x8_data)[5]),                     \
        fp16_ieee_to_fp32_value((fp16x8_data)[6]),                     \
        fp16_ieee_to_fp32_value((fp16x8_data)[7])                      \
    }

/**
 * Converts each element of a 4-wide vector of single-precision FP values to
 * half-precision FP, using the FP16 library.
 */
#define _SW_CVT_PS_PH_128(fp32x4_data, rounding_mode)                  \
    {                                                                  \
        fp16_ieee_from_fp32_value((fp32x4_data)[0]),                   \
        fp16_ieee_from_fp32_value((fp32x4_data)[1]),                   \
        fp16_ieee_from_fp32_value((fp32x4_data)[2]),                   \
        fp16_ieee_from_fp32_value((fp32x4_data)[3])                    \
    }

/**
 * Converts each element of a 4-wide vector of half-precision FP values to
 * single-precision FP, using the FP16 library.
 */
#define _SW_CVT_PH_PS_128(fp16x4_data)                                 \
    {                                                                  \
        fp16_ieee_to_fp32_value((fp16x4_data)[0]),                     \
        fp16_ieee_to_fp32_value((fp16x4_data)[1]),                     \
        fp16_ieee_to_fp32_value((fp16x4_data)[2]),                     \
        fp16_ieee_to_fp32_value((fp16x4_data)[3])                      \
    }

//=----------------- Manual assembly implementions -------------------=//
//
// These are used if certain side-effects of adding -mf16c are unacceptable
// (for example, we want to enable F16C without AVX).

/**
 * Converts each element of a 4-wide vector of single-precision FP values to
 * half-precision FP using the x86 instruction vctps2ph. Assumes the presence
 * of the FP16 conversion ISA extension on the target machine.
 *
 * @param a A vector of 4 single-precision floats (e.g. v4fp_t).
 * @param imm8 The rounding mode.
 */
static inline __m128i __smaug_vcvtps2ph(__m128i a, int imm8) {
    __m128i res = (__m128i){ 0 };
    __asm__ volatile("vcvtps2ph %2, %1, %0" : "+xm"(res) : "x"(a), "i"(imm8) :);
    return res;
}

/**
 * Converts each element of a 4-wide vector of half-precision FP values to
 * single-precision FP using the x86 instruction vctph2ps. Assumes the presence
 * of the FP16 conversion ISA extension on the target machine.
 *
 * @param a A vector of 4 half-precision floats (e.g. v4ph_t).
 * @param imm8 The rounding mode.
 */
static inline __m128i __smaug_vcvtph2ps(__m128i a) {
    __m128i res = (__m128i){ 0 };
    __asm__ volatile("vcvtph2ps %1, %0" : "+x"(res) : "xm"(a) :);
    return res;
}

/**
 * Converts each element of an 8-wide vector of single-precision FP values to
 * half-precision FP using the x86 instruction vctps2ph. Assumes the presence
 * of the FP16 conversion ISA extension on the target machine.
 *
 * @param a A vector of 8 single-precision floats (e.g. v8fp_t).
 * @param imm8 The rounding mode.
 */
typedef float __smaug256 __attribute__((__vector_size__(32)));
static inline __m128i __smaug_vcvtps2ph256(__smaug256 a, int imm8) {
    __m128i res = (__m128i){ 0 };
    __asm__ volatile("vcvtps2ph %2, %1, %0" : "+xm"(res) : "x"(a), "i"(imm8) :);
    return res;
}

/**
 * Converts each element of a 4-wide vector of half-precision FP values to
 * single-precision FP using the x86 instruction vctph2ps. Assumes the presence
 * of the FP16 conversion ISA extension.
 *
 * @param a A vector of 8 half-precision floats (e.g. v8ph_t).
 * @param imm8 The rounding mode.
 */
static inline __smaug256 __smaug_vcvtph2ps256(__m128i a) {
    __smaug256 res = (__smaug256){ 0 };
    __asm__ volatile("vcvtph2ps %1, %0" : "+x"(res) : "xm"(a) :);
    return res;
}

//=----------------- 128-bit conversion instructions -----------------=//

/** \defgroup VectorFPConversionMacros
 *
 * Use these macros to portably convert between vectors of single-precision and
 * half-precision floats, instead of directly calling the FP16 library
 * functions or the manual asm implementations. Depending on the compilation
 * mode (e.g. native vs gem5 vs LLVM-Tracer) and hardware support, different
 * underlying implementations will be selected.
 *
 * _CVT_PS_PH_128: Convert 4-wide vectors from single to half precision.
 * _CVT_PH_PS_128: Convert 4-wide vectors from half to single precision.
 * _CVT_PS_PH_256: Convert 8-wide vectors from single to half precision.
 * _CVT_PH_PS_256: Convert 8-wide vectors from half to single precision.
 * @{
 */

#ifdef __F16C__

// Use built-in compiler intrinsics.
#define _CVT_PS_PH_128(p4_fp32_data, rounding_mode)                    \
    _mm_cvtps_ph(p4_fp32_data, rounding_mode)
#define _CVT_PH_PS_128(p4_fp16_data) _mm_cvtph_ps(p4_fp16_data)

#elif defined(__USE_F16C_ANYWAYS__)

// We can't use the compiler intrinsics, so use our own manual asm.
#define _CVT_PS_PH_128(p4_fp32_data, rounding_mode)                    \
    __smaug_vcvtps2ph(p4_fp32_data, rounding_mode)
#define _CVT_PH_PS_128(p4_fp16_data) __smaug_vcvtph2ps(p4_fp16_data)

#else

#ifdef TRACE_MODE
#  warning "No F16C: LLVM-Tracer cannot emit IR FP convert instructions!"
#endif

// Fallback to the SW emulations.
#define _CVT_PS_PH_128(p4_fp32_data, rounding_mode)                    \
    _SW_CVT_PS_PH_128(p4_fp32_data, rounding_mode)
#define _CVT_PH_PS_128(p4_fp16_data) _SW_CVT_PH_PS_128(p4_fp16_data)

#endif  // __F16C__

//=----------------- 256-bit conversion instructions -----------------=//

// gem5 doesn't support the 256-bit iforms, due to lack of support for YMM
// registers, so fallback to the SW.
#if defined(GEM5)

#define _CVT_PS_PH_256(p8_fp32_data, rounding_mode)                    \
    _SW_CVT_PS_PH_256(p8_fp32_data, rounding_mode)
#define _CVT_PH_PS_256(p8_fp16_data) _SW_CVT_PH_PS_256(p8_fp16_data)

#elif defined(__F16C__)

#define _CVT_PS_PH_256(p8_fp32_data, rounding_mode)                    \
    _mm256_cvtps_ph(p8_fp32_data, rounding_mode)
#define _CVT_PH_PS_256(p8_fp16_data) _mm256_cvtph_ps(p8_fp16_data)

#elif defined(__USE_F16C_ANYWAYS__)

#define _CVT_PS_PH_256(p8_fp32_data, rounding_mode)                    \
    __smaug_vcvtps2ph256(p8_fp32_data, rounding_mode)
#define _CVT_PH_PS_256(p8_fp16_data) __smaug_vcvtph2ps256(p8_fp16_data)

#else

#ifdef TRACE_MODE
#  warning "No F16C: LLVM-Tracer cannot emit IR FP convert instructions!"
#endif

// No F16C in HW; fallback to SW implementations.
#define _CVT_PS_PH_256(p8_fp32_data, rounding_mode)                    \
    _SW_CVT_PS_PH_256(p8_fp32_data, rounding_mode)
#define _CVT_PH_PS_256(p8_fp16_data) _SW_CVT_PH_PS_256(p8_fp16_data)

#endif

/**
 * @}
 */

//=----------------- Scalar conversion instructions -----------------=//

/**
 * Converts a scalar float from single to half precision.
 *
 * Although ISA support exists for these instructions in F16C, LLVM did not add
 * support for this intrinsic until 3.8, so we need to fallback to the lowest
 * common denominator.
 *
 * TODO(xyzsam): Update this implementation now that we run on LLVM 6.
 */
#define _CVT_SS_SH(val, rounding_mode) fp16_ieee_from_fp32_value(val)

/** Converts a scalar float from half to single precision. */
#define _CVT_SH_SS(val) fp16_ieee_to_fp32_value(val)

//=----------------- Miscellaneous vector instructions -----------------=//

/** Shuffle 64-bit chunks of two XMM registers based on value of imm8. */
#define _SHUFFLE_PD(a, b, imm8) _mm_shuffle_pd((__m128d)(a), (__m128d)(b), imm8)

#endif
