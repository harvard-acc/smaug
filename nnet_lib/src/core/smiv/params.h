#ifndef _SMIV_PARAMS_H_
#define _SMIV_PARAMS_H_

#include <stdint.h>

#define DATAPATH_WIDTH 4
#define SHIFT_REG_SIZE 16
#define MAX_BATCH 8

#ifndef VECTOR_SIZE
#define VECTOR_SIZE 8
#elif VECTOR_SIZE != 8
#error "Existing VECTOR_SIZE is incompatible with SMIV!"
#endif

// Scalar types.
typedef float fp_t;
typedef int sfx_t;
typedef unsigned ufx_t;
typedef uint16_t fp16_t;

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

#define VEC_ARRAY_1D(type, output_name, input_name)                            \
    type* output_name = (type*)(input_name)

#define VEC_ARRAY_2D(type, output_name, input_name, cols)                      \
    type(*output_name)[(cols) / (VECTOR_SIZE)] =                               \
            (type(*)[(cols) / (VECTOR_SIZE)])input_name

#define VEC_ARRAY_3D(type, output_name, input_name, rows, cols)                \
    type(*output_name)[(rows)][(cols) / (VECTOR_SIZE)] =                       \
            (type(*)[(rows)][(cols) / (VECTOR_SIZE)])input_name

#define VEC_ARRAY_4D(type, output_name, input_name, height, rows, cols)        \
    type(*output_name)[(height)][(rows)][(cols) / (VECTOR_SIZE)] =             \
            (type(*)[(height)][(rows)][(cols) / (VECTOR_SIZE)])input_name

// Apply a mask to a 256-bit packed single precision FP vector.
//
// The mask is a vector of either 0s or -1s (all 1s). Entries that are have a
// mask of 0 are zeroed out.
//
// LLVM is smart enough to turn this into a SELECT instruction, rather than a
// bitwise mask!
//
// Args:
//    input: a v8fp_t vector
//    mask: a v8sfx_t vector of either 0s or -1s.
#define VEC256_MASK(input, mask) ((v8fp_t)((v8sfx_t)input & mask))

// Same as above, but for 128-bit vectors.
#define VEC128_MASK(input, mask) ((v4fp_t)((v4sfx_t)input & mask))

#endif
