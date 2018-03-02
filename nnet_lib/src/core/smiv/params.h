#ifndef _SMIV_PARAMS_H_
#define _SMIV_PARAMS_H_

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

// 8 packed 32-bit floating point values.
typedef fp_t v8fp_t
        __attribute__((__vector_size__(VECTOR_SIZE * sizeof(fp_t))));
// 4 packed 32-bit floating point values.
typedef fp_t v4fp_t
        __attribute__((__vector_size__(VECTOR_SIZE / 2 * sizeof(fp_t))));
// 16 packed 32-bit floating point values.
typedef short v16fp_t
        __attribute__((__vector_size__(VECTOR_SIZE * 2 * sizeof(fp_t))));
// 16 packed 16-bit floating point values.
typedef short v16ph_t
        __attribute__((__vector_size__(VECTOR_SIZE * 2 * sizeof(short))));
// 8 packed 16-bit floating point values.
typedef short v8ph_t
        __attribute__((__vector_size__(VECTOR_SIZE * sizeof(short))));
// 8 packed 32-bit integer values.
typedef sfx_t v8sfx_t
        __attribute__((__vector_size__(VECTOR_SIZE * sizeof(sfx_t))));

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

// Apply a mask to a vector literal.
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
#define VEC_MASK(input, mask) ((v8fp_t)((v8sfx_t)input & mask))

#endif
