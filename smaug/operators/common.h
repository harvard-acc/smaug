/**
 * \file common.h
 * \brief Utilities for writing and invoking Aladdin kernels from Operators.
 */

#ifndef _OPERATORS_COMMON_H_
#define _OPERATORS_COMMON_H_

#include <stdint.h>

#include "gem5/sampling_interface.h"

#ifdef DMA_MODE
#ifdef __cplusplus
extern "C" {
#endif
#include "gem5/dma_interface.h"
#ifdef __cplusplus
}
#endif
#include "gem5/aladdin_sys_connection.h"
#include "gem5/aladdin_sys_constants.h"
#include "gem5/systolic_array_connection.h"
#endif

#ifdef __cplusplus
// Functions for invoking kernels and mapping arrays.
//
// If gem5 simulation is not used, the pure software version of the accelerate
// kernels will be invoked.
//
// These functions should be called from C++ files and not be included in C
// files.

#include <string>
#include <utility>
#include <memory>
#include "smaug/core/globals.h"
#include "tracer/trace_logger_aladdin.h"

namespace smaug {

/**
 * Return the name of the dynamic trace for this accelerator.
 *
 * @param accelIdx The ID of this accelerator.
 */
std::string getTraceName(int accelIdx);

/**
 * The generic blocking interface for all accelerator kernel functions.
 *
 * All accelerated kernels should be called via this interface, and different
 * things will happen based on how the program is being run:
 *
 * - As a native binary: the kernel function is directly called.
 * - As an LLVM-Tracer instrumented binary: sets the file name of the dynamic
 *   trace being generated, then calls the kernel function.
 * - In gem5-Aladdin: invokes the Aladdin model of the specified accelerator.
 *
 * This is a blocking call: in gem5-Aladdin mode, the thread will wait until
 * the accelerator finishes. For a non-blocking call, use invokeKernelNoBlock.
 *
 * @param accelIdx Setes the suffix of the dynamic trace to XXX_acc[accelIdx].
 * Used if you want to generate multiple independent traces to simulate
 * multiple accelerators.
 * @param reqCode The ID of the accelerator to invoke.
 * @param kernel The kernel function to invoke in native/LLVM-Tracer mode.
 * @param args The arguments to the kernel function.
 */
template <typename Kernel, typename... Args>
void invokeKernel(int accelIdx,
                  unsigned reqCode,
                  const Kernel& kernel,
                  Args&&... args) {
    if (runningInSimulation) {
        invokeAcceleratorAndBlock(reqCode);
    } else {
#ifdef TRACE_MODE
        llvmtracer_set_trace_name(getTraceName(accelIdx).c_str());
#endif
        kernel(std::forward<Args>(args)...);
    }
}

/**
 * A generic interface for all accelerator kernel functions.
 *
 * This is a convenience function that sets accelIdx = 0, so only one dynamic
 * trace file will be generated.
 */
template <typename Kernel, typename... Args>
void invokeKernel(unsigned reqCode, const Kernel& kernel, Args&&... args) {
    invokeKernel(0, reqCode, kernel, std::forward<Args>(args)...);
}

/**
 * A generic non-blocking interface to accelerated kernel functions.
 *
 * The only difference between this and invokeKernel is that in gem5-Aladdin
 * mode, the thread will start Aladdin and then return immediately. The calling
 * thread is responsible for checking the status of the accelerator and taking
 * action appropriately.
 */
template <typename Kernel, typename... Args>
std::unique_ptr<volatile int> invokeKernelNoBlock(int accelIdx,
                                                  unsigned reqCode,
                                                  const Kernel& kernel,
                                                  Args&&... args) {
    if (runningInSimulation) {
        return std::unique_ptr<volatile int>(
                invokeAcceleratorAndReturn(reqCode));
    } else {
#ifdef TRACE_MODE
        llvmtracer_set_trace_name(getTraceName(accelIdx).c_str());
#endif
        kernel(std::forward<Args>(args)...);
        return nullptr;
    }
}

/**
 * Maps an array of data to the accelerator.
 *
 * This enables the accelerator to access host memory via DMA or caching memory
 * accesses.
 *
 * @param reqCode The ID of the accelerator
 * @param arrayName The name of the array as it appears in the top-level
 * accelerator function signature.
 * @param baseAddr The base address of the array (e.g. &array[0]).
 * @param size The size of the array.
 */
void mapArrayToAccel(unsigned reqCode,
                     const char* arrayName,
                     void* baseAddr,
                     size_t size);

/**
 * Sets what memory access mechanism the accelerator will use when accessing
 * this array.
 *
 * This lets the user decide at runtime whether to access a hots array over
 * DMA, hardware caching, or ACP.
 *
 * @param reqCode The ID of the accelerator
 * @param arrayName The name of the array as it appears in the accelerator's
 * function signature.
 * @param memType The memory access mechanism.
 */
void setArrayMemTypeIfSimulating(unsigned reqCode,
                                 const char* arrayName,
                                 MemoryType memType);

}  // namespace smaug
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Returns the smallest multiple of align that is >= request. */
size_t next_multiple(size_t request, size_t align);

#ifdef __cplusplus
}
#endif

/**
 * The activation function to apply to an operator's output in hardware.
 *
 * This is a struct that bridges the C++ class ActivationInfo with the C kernel.
 */
typedef enum _activation_type {
    NO_ACTIVATION,
    RELU,
    RELU_THRESHOLD,
    LRELU,
    ELU,
    SELU,
    TANH,
    HARD_TANH,
    SIGMOID,
    SOFTMAX
} activation_type;

/**
 *
 * Parameters to the activation function hardware.
 *
 * This is a struct that bridges the C++ class ActivationInfo with the C kernel.
 */
typedef struct _activation_param_t {
    // LReLU
    float slope;
    // ELU/SELU
    float alpha;
    float lambda;
    // Hard Tanh
    float min;
    float max;
} activation_param_t;

#ifdef __cplusplus

/**
 * Specifies an activation function and relevant parameters.
 */
struct ActivationInfo {
   public:
    ActivationInfo() : function(activation_type::NO_ACTIVATION) {}
    ActivationInfo(activation_type _function) : function(_function) {
        // Use default parameters if not specified.
        switch (_function) {
            case activation_type::LRELU:
                params.slope = 0.2;
                break;
            case activation_type::ELU:
                params.alpha = 0.1;
                break;
            case activation_type::SELU:
                params.alpha = 1.6733;
                params.lambda = 1.0507;
                break;
            case activation_type::HARD_TANH:
                params.min = -1;
                params.max = 1;
                break;
            default:
                break;
        }
    }
    ActivationInfo(activation_type _function, activation_param_t _params)
            : function(_function), params(_params) {}
    activation_type function;
    activation_param_t params;
};
#endif

/**
 * Levels of simulation sampling to apply to certain accelerator kernels.
 *
 * It is the responsibility of the kernel writer to translate this qualitative
 * description into a kernel-appropriate modification of the code behavior. For
 * example, Low could mean "only sample the innermost loop", while VeryHigh
 * could be translated to "sample all loop levels".  Kernels do not need to
 * support every single sampling level.
 */
typedef enum _SamplingLevel {
    NoSampling = 0,
    Low = 1,
    Medium = 2,
    High = 3,
    VeryHigh = 4
} SamplingLevel;

/**
 * Simulation sampling information maintained by the Operator and passed to the
 * accelerated kernel.
 */
typedef struct _SamplingInfo {
    /** Qualitative level of sampling. */
    SamplingLevel level;
    /**
     * The requested number of iterations to run a sampled loop. Depending on
     * the kernel's restrictions, the actual iterations run could be greater.
     */
    int num_sample_iterations;
} SamplingInfo;

// Scalar types.
typedef float fp_t;
typedef int sfx_t;
typedef unsigned ufx_t;
typedef uint16_t fp16_t;
typedef uint16_t float16;

#define CACHELINE_SIZE 32
#define LOG_PAGE_SIZE 12

/** \defgroup VectorData Working with SIMD in C.
 *
 * Typedefs and macros for working with vector data.
 *
 * @{
 */

/**
 * Vector size used in SMV backends.
 */
#ifndef VECTOR_SIZE
#define VECTOR_SIZE 8
#endif

/** 16 packed 32-bit floating point values. */
typedef fp16_t v16fp_t
        __attribute__((__vector_size__(VECTOR_SIZE * 2 * sizeof(fp_t))));
/** 8 packed 32-bit floating point values. */
typedef fp_t v8fp_t
        __attribute__((__vector_size__(VECTOR_SIZE * sizeof(fp_t))));
/** 4 packed 32-bit floating point values. */
typedef fp_t v4fp_t
        __attribute__((__vector_size__(VECTOR_SIZE / 2 * sizeof(fp_t))));

/** 16 packed 16-bit floating point values. */
typedef fp16_t v16ph_t
        __attribute__((__vector_size__(VECTOR_SIZE * 2 * sizeof(fp16_t))));
/**  8 packed 16-bit floating point values. */
typedef fp16_t v8ph_t
        __attribute__((__vector_size__(VECTOR_SIZE * sizeof(fp16_t))));
/** 4 packed 16-bit floating point values. */
typedef fp16_t v4ph_t
        __attribute__((__vector_size__(VECTOR_SIZE / 2 * sizeof(fp16_t))));

/** 8 packed 32-bit integer values. */
typedef sfx_t v8sfx_t
        __attribute__((__vector_size__(VECTOR_SIZE * sizeof(sfx_t))));
/** 4 packed 32-bit integer values. */
typedef sfx_t v4sfx_t
        __attribute__((__vector_size__(VECTOR_SIZE / 2 * sizeof(sfx_t))));

/** 8 packed 8-bit bool values. */
typedef uint8_t v8bl_t
        __attribute__((__vector_size__(VECTOR_SIZE * sizeof(uint8_t))));

/**
 * Apply a elementwise mask to a 128-bit packed single precision FP vector.
 *
 * The mask is a vector of either 0s or -1s (all 1s). Entries that are have a
 * mask of 0 are zeroed out.
 *
 * LLVM is smart enough to turn this into a SELECT instruction, rather than a
 * bitwise mask!
 *
 * @param input A v4fp_t vector.
 * @param mask A v4sfx_t vector of either 0s or -1s.
 */
#define VEC128_MASK(input, mask) ((v4fp_t)((v4sfx_t)input & mask))

/**
 * Same as VEC128_MASK, but for 256-bit vectors.
 *
 * @param input A v4fp_t vector.
 * @param mask A v4sfx_t vector of either 0s or -1s.
 */
#define VEC256_MASK(input, mask) ((v8fp_t)((v8sfx_t)input & mask))

/**
 * @}
 */

/** \defgroup MultiDimArrays Multi-dim arrays in C
 *
 * Use these convenience macros to cast a raw pointer into a multidimensional
 * variable-length array, which lets us use `[]` notation instead of manually
 * linearizing the index.
 *
 * Usage: Suppose we have an `int* array` pointer.
 *
 * To convert this into a multidimensional array, call the appropriate macro,
 * providing the type, a new variable name (often the same as the original with
 * an underscore prefixed), and then listing out all but the first dimension.
 *
 * For example:
 *
 *    ```c
 *    // To convert into a 5x4 array:
 *    ARRAY_2D(int, _array, array, 4);
 *    _array[0][1] = 1;
 *
 *    // To convert into a 5x4x3 array:
 *    ARRAY_3D(int, _array, array, 4, 3);
 *    _array[0][1][2] = 1;
 *
 *    // To convert into a 5x4x3x2 array:
 *    ARRAY_4D(int, _array, array, 4, 3, 2);
 *    _array[0][1][2][3] = 1;
 *    ```
 * @{
 */

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

#define VEC_ARRAY_2D(TYPE, output_array_name, input_array_name, cols)          \
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

/** @} */

#endif


/** \defgroup AladdinHelpers Utilities for writing Aladdin kernels.
 *
 * Macros to assist in writing code for Aladdin/LLVM-Tracer that translates
 * into an efficient hardware model.
 *
 * @{
 */

/** \defgroup AladdinMath Common math functions in Aladdin
 *
 * Macros for computing the min/max of a group of elements.
 *
 * Why macros and not functions (or a loop)? A loop takes O(n) cycles to
 * compute the maximum, when it could be done in O(log n) time with a tree
 * based implementation. But Aladdin regards function calls as a hard
 * dependency that it does not optimize across, so we would not get the
 * parallelism we expect from the tree. Thus, these are best expressed as
 * macros.
 *
 * @{
 */
#define max2(A, B) (((A) > (B)) ? (A) : (B))
#define max3(e0, e1, e2) max2(max2(e0, e1), e2)
#define max4(e0, e1, e2, e3) max2(max2(e0, e1), max2(e2, e3))
#define max8(e0, e1, e2, e3, e4, e5, e6, e7)                                   \
    max2(max4(e0, e1, e2, e3), max4(e4, e5, e6, e7))
#define max9(e0, e1, e2, e3, e4, e5, e6, e7, e8)                               \
    max2(max8(e0, e1, e2, e3, e4, e5, e6, e7), e8)
#define min2(A, B) (((A) < (B)) ? (A) : (B))

/** Implements the ceiling function of A/B. */
#define FRAC_CEIL(A, B) ((A) / (B) + ((A) % (B) != 0))
/**
 * @}
 */

/**
 * We have to disable all function inlining at the global level for Aladdin +
 * LLVM-Tracer to work, but sometimes we do want to force inline functions
 * (otherwise we run into all the issues of function call barriers in Aladdin).
 * Add ALWAYS_INLINE before the function declaration to force inlining on this
 * function. Only add this on instrumented functions; it's usually unnecessary
 * and often generates a lot of compiler warnings.
 */
#ifdef TRACE_MODE
#define ALWAYS_INLINE __attribute__((__always_inline__))
#else
#define ALWAYS_INLINE
#endif

/**
 * An assertion macro which disables asserts in LLVM-Tracer instrumented code.
 */
#ifdef TRACE_MODE
#define ASSERT(x)
#else
#define ASSERT(x) assert(x)
#endif

/**
 * Tell the compiler to assume a pointer is aligned on some byte boundary. This
 * is not supported in clang 3.4.
 */
#ifdef TRACE_MODE
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#else
#define ASSUME_ALIGNED(ptr, args...) __builtin_assume_aligned((ptr), args)
#endif

#define MAYBE_UNUSED __attribute__((__unused__))

/**
 * @}
 */

#endif
