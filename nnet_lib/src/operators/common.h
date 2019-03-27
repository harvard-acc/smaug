#ifndef _OPERATORS_COMMON_H_
#define _OPERATORS_COMMON_H_

#include <stdint.h>

#ifdef DMA_MODE
#ifdef __cplusplus
extern "C" {
#endif
#include "dma_interface.h"
#ifdef __cplusplus
}
#endif
#include "gem5/aladdin_sys_connection.h"
#include "gem5/aladdin_sys_constants.h"
#endif

// Scalar types.
typedef float fp_t;
typedef int sfx_t;
typedef unsigned ufx_t;
typedef uint16_t fp16_t;

#ifndef VECTOR_SIZE
#define VECTOR_SIZE 8
#endif

// 16 packed 32-bit floating point values.
typedef fp16_t v16fp_t
        __attribute__((__vector_size__(VECTOR_SIZE * 2 * sizeof(fp_t))));
// 8 packed 32-bit floating point values.
typedef fp_t v8fp_t
        __attribute__((__vector_size__(VECTOR_SIZE * sizeof(fp_t))));
// 4 packed 32-bit floating point values.
typedef fp_t v4fp_t
        __attribute__((__vector_size__(VECTOR_SIZE / 2 * sizeof(fp_t))));

// 16 packed 16-bit floating point values.
typedef fp16_t v16ph_t
        __attribute__((__vector_size__(VECTOR_SIZE * 2 * sizeof(fp16_t))));
// 8 packed 16-bit floating point values.
typedef fp16_t v8ph_t
        __attribute__((__vector_size__(VECTOR_SIZE * sizeof(fp16_t))));
// 4 packed 16-bit floating point values.
typedef fp16_t v4ph_t
        __attribute__((__vector_size__(VECTOR_SIZE / 2 * sizeof(fp16_t))));

// 8 packed 32-bit integer values.
typedef sfx_t v8sfx_t
        __attribute__((__vector_size__(VECTOR_SIZE * sizeof(sfx_t))));
typedef sfx_t v4sfx_t
        __attribute__((__vector_size__(VECTOR_SIZE / 2 * sizeof(sfx_t))));

// Use these convenience macros to cast a raw pointer into a multidimensional
// variable-length array, which lets us use [] notation inside of the ugly
// sub2ind syntax!
//
// Usage:
//   If we have an array like array[5][4]:
//      ARRAY_2D(TYPE, output_name, array, 4);
//
//   If we have an array like array[5][4][3]:
//      ARRAY_3D(TYPE, output_name, array, 4, 3);
//
//   If we have an array like array[5][4][3][2]
//      ARRAY_4D(TYPE, output_name, array, 4, 3, 2);
//
//   And so on...

#if defined(__clang__)

#define TO_TYPE(output_array_name, input_array_name)                           \
    output_array_name##_t output_array_name =                                  \
            (output_array_name##_t)(input_array_name)

#define ARRAY_1D(TYPE, output_array_name, input_array_name)                    \
    TYPE* output_array_name = (TYPE*)input_array_name

#define ARRAY_2D(TYPE, output_array_name, input_array_name, DIM_1)             \
    typedef TYPE(*output_array_name##_t)[DIM_1];                               \
    TO_TYPE(output_array_name, input_array_name)

#define ARRAY_3D(TYPE, output_array_name, input_array_name, DIM_1, DIM_2)      \
    typedef TYPE(*output_array_name##_t)[DIM_1][DIM_2];                        \
    TO_TYPE(output_array_name, input_array_name)

#define ARRAY_4D(                                                              \
        TYPE, output_array_name, input_array_name, DIM_1, DIM_2, DIM_3)        \
    typedef TYPE(*output_array_name##_t)[DIM_1][DIM_2][DIM_3];                 \
    TO_TYPE(output_array_name, input_array_name)

#define ARRAY_5D(                                                              \
        TYPE, output_array_name, input_array_name, DIM_1, DIM_2, DIM_3, DIM_4) \
    typedef TYPE(*output_array_name##_t)[DIM_1][DIM_2][DIM_3][DIM_4];          \
    TO_TYPE(output_array_name, input_array_name)

#define VEC_ARRAY_1D(TYPE, output_array_name, input_array_name)                \
    TYPE* output_array_name = (TYPE*)input_array_name

#define VEC_ARRAY_2D(TYPE, output_array_name, input_array_name)                \
    typedef TYPE(*output_array_name##_t)[(cols) / VECTOR_SIZE];                \
    TO_TYPE(output_array_name, input_array_name)

#define VEC_ARRAY_3D(TYPE, output_array_name, input_array_name, rows, cols)    \
    typedef TYPE(*output_array_name##_t)[(rows)][(cols) / VECTOR_SIZE];        \
    TO_TYPE(output_array_name, input_array_name)

#define VEC_ARRAY_4D(                                                          \
        TYPE, output_array_name, input_array_name, height, rows, cols)         \
    typedef TYPE(                                                              \
            *output_array_name##_t)[(height)][(rows)][(cols) / VECTOR_SIZE];   \
    TO_TYPE(output_array_name, input_array_name)

#elif defined(__GNUC__)

#define ARRAY_1D(TYPE, output_array_name, input_array_name)                    \
    TYPE* output_array_name = (TYPE*)input_array_name

#define ARRAY_2D(TYPE, output_array_name, input_array_name, DIM_1)             \
    TYPE(*output_array_name)[DIM_1] = (TYPE(*)[DIM_1])input_array_name

#define ARRAY_3D(TYPE, output_array_name, input_array_name, DIM_1, DIM_2)      \
    TYPE(*output_array_name)[DIM_1][DIM_2] =                                   \
        (TYPE(*)[DIM_1][DIM_2])input_array_name

#define ARRAY_4D(                                                              \
    TYPE, output_array_name, input_array_name, DIM_1, DIM_2, DIM_3)            \
        TYPE(*output_array_name)[DIM_1][DIM_2][DIM_3] =                        \
            (TYPE(*)[DIM_1][DIM_2][DIM_3])input_array_name

#define ARRAY_5D(                                                              \
    TYPE, output_array_name, input_array_name, DIM_1, DIM_2, DIM_3, DIM_4)     \
        TYPE(*output_array_name)[DIM_1][DIM_2][DIM_3][DIM_4] =                 \
            (TYPE(*)[DIM_1][DIM_2][DIM_3][DIM_4])input_array_name

#define VEC_ARRAY_1D(TYPE, output_array_name, input_array_name)                \
    TYPE* output_array_name = (TYPE*)(input_array_name)

#define VEC_ARRAY_2D(TYPE, output_array_name, input_array_name, cols)          \
    TYPE(*output_array_name)                                                   \
    [(cols) / (VECTOR_SIZE)] =                                                 \
            (TYPE(*)[(cols) / (VECTOR_SIZE)]) input_array_name

#define VEC_ARRAY_3D(TYPE, output_array_name, input_array_name, rows, cols)    \
    TYPE(*output_array_name)                                                   \
    [(rows)][(cols) / (VECTOR_SIZE)] =                                         \
            (TYPE(*)[(rows)][(cols) / (VECTOR_SIZE)]) input_array_name

#define VEC_ARRAY_4D(                                                          \
        TYPE, output_array_name, input_array_name, height, rows, cols)         \
    TYPE(*output_array_name)                                                   \
    [(height)][(rows)][(cols) / (VECTOR_SIZE)] =                               \
            (TYPE(*)[(height)][(rows)][(cols) / (VECTOR_SIZE)])                \
                    input_array_name

#endif

#define STRING(arg) #arg

// This is to avoid a ton of spurious unused variable warnings when
// we're not building for gem5.
#define UNUSED(x) (void)(x)

// Convenience macros to switch between invoking an accelerator (if building a
// binary for gem5) or just calling the kernel function in software.
//
// Usage:
//
//  These macros expand differently based on whether the GEM5_HARNESS macro is
//  defined. If so, then this binary is meant to be run under gem5, invoking
//  accelerators; if not, this binary should run the pure software version of
//  the accelerated kernels.
//
//  If GEM5_HARNESS is defined:
//
//     MAP_ARRAY_TO_ACCEL(myReqCode, myArrayName, myArrayPtr, mySize)
//        ===>   mapArrayToAccelerator(myReqCode, myArrayName, myArrayPtr, mySize)
//
//     INVOKE_KERNEL(myReqCode, kernelFuncName, args...)
//        ===>   invokeAcceleratorAndBlock(myReqCode)
//
//  Otherwise:
//     MAP_ARRAY_TO_ACCEL(myReqCode, myArrayName, myArrayPtr, mySize)
//        expands to nothing
//
//     INVOKE_KERNEL(myReqCode, kernelFuncName, args...)
//        ===>  kernelFuncName(args)
//
#ifdef GEM5_HARNESS

#define MAP_ARRAY_TO_ACCEL(req_code, name, base_addr, size)                    \
    mapArrayToAccelerator(req_code, name, base_addr, size)
#define INVOKE_KERNEL(req_code, kernel_ptr, args...)                           \
    do {                                                                       \
        UNUSED(kernel_ptr);                                                    \
        invokeAcceleratorAndBlock(req_code);                                   \
    } while (0)
#define INVOKE_KERNEL_NOBLOCK(req_code, finish_flag, kernel_ptr, args...)      \
    do {                                                                       \
        UNUSED(kernel_ptr);                                                    \
        invokeAcceleratorAndReturn2(req_code, finish_flag);                    \
    } while (0)

#else

#define MAP_ARRAY_TO_ACCEL(req_code, name, base_addr, size)                    \
    do {                                                                       \
        UNUSED(req_code);                                                      \
        UNUSED(name);                                                          \
        UNUSED(base_addr);                                                     \
        UNUSED(size);                                                          \
    } while (0)
#define INVOKE_KERNEL(req_code, kernel_ptr, args...) kernel_ptr(args)
#define INVOKE_KERNEL_NOBLOCK(req_code, finish_flag, kernel_ptr, args...)      \
    kernel_ptr(args)

#endif

// Simplified version of MAP_ARRAY_TO_ACCEL.
//
// This assumes that the current name of the base pointer is also the name of
// the array in the top level function of the dynamic trace. THIS IS VERY
// IMPORTANT - if the argument passed to a top level function has been renamed in
// the function, then this WILL NOT WORK!
//
// MAP_ARRAY(myReqCode, myArray, mySize)
//    ===>   MAP_ARRAY_TO_ACCEL(myReqCode, "myArray", myArray, mySize)
#define MAP_ARRAY(req_code, name_and_base_addr, size)                          \
    MAP_ARRAY_TO_ACCEL(                                                        \
            req_code, STRING(name_and_base_addr), name_and_base_addr, size)


// Macros for computing the maximum of a group of elements.
//
// Why macros and not functions (or a loop)? A loop takes O(n) cycles to
// compute the maximum, when it could be done in O(log n) time with a tree
// based implementation. But Aladdin regards function calls as a hard
// dependency that it does not optimize across, so we would not get the
// parallelism we expect from the tree. Thus, these must be macros.
//
// I've only implemented a few of these. These are only meant for the pooling
// layers, and we shouldn't need more than a 3x3 pooling layer anyways.
#define max2(A, B) (((A) > (B)) ? (A) : (B))
#define max3(e0, e1, e2) max2(max2(e0, e1), e2)
#define max4(e0, e1, e2, e3) max2(max2(e0, e1), max2(e2, e3))
#define max8(e0, e1, e2, e3, e4, e5, e6, e7)                                   \
    max2(max4(e0, e1, e2, e3), max4(e4, e5, e6, e7))
#define max9(e0, e1, e2, e3, e4, e5, e6, e7, e8)                               \
    max2(max8(e0, e1, e2, e3, e4, e5, e6, e7), e8)
#define min2(A, B) (((A) < (B)) ? (A) : (B))

#define FRAC_CEIL(A, B) ((A) / (B) + ((A) % (B) != 0))

// Compiler-specific features.
//
// ALWAYS_INLINE:
// We have to disable all function inlining at the global level for Aladdin +
// LLVM-Tracer to work, but sometimes we do want to force inline functions
// (otherwise we run into all the issues of function call barriers in Aladdin).
// Add ALWAYS_INLINE before the function declaration to force inlining on this
// function.  Don't do this except when we're tracing though; usually it is not
// necessary and it generates a lot of compiler warnings.
//
// ASSERT:
// Disable asserts within instrumented when tracing.
//
// ASSUME_ALIGNED:
// Tell the compiler to assume a pointer is aligned on some byte boundary. This
// is not supported in clang 3.4.
#ifdef TRACE_MODE
#define ALWAYS_INLINE __attribute__((__always_inline__))
#define ASSERT(x)
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#else
#define ALWAYS_INLINE
#define ASSERT(x) assert(x)
#define ASSUME_ALIGNED(ptr, args...) __builtin_assume_aligned((ptr), args)
#endif

#define MAYBE_UNUSED __attribute__((__unused__))

#endif
