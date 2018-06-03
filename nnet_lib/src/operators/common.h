#ifndef _OPERATORS_COMMON_H_
#define _OPERATORS_COMMON_H_

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

#endif

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
