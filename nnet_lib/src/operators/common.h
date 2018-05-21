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

#endif
