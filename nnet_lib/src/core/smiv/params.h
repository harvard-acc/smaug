#ifndef _SMIV_PARAMS_H_
#define _SMIV_PARAMS_H_

#define DATAPATH_WIDTH 4
#define SHIFT_REG_SIZE 16
#define MAX_BATCH 8
#define VECTOR_SIZE 8

// A 4 byte value type.
typedef float fp_t;
// Vector of 8 scalar values.
typedef fp_t v8fp_t
        __attribute__((__vector_size__(VECTOR_SIZE * sizeof(fp_t))));

#define VEC_ARRAY_2D(TYPE, OUTPUT_NAME, INPUT_NAME, INPUT_WIDTH) \
    TYPE (*OUTPUT_NAME)[(INPUT_WIDTH)/(VECTOR_SIZE)] =           \
        (TYPE(*)[(INPUT_WIDTH)/(VECTOR_SIZE)])INPUT_NAME


#endif
