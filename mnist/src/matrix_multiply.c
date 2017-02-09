#include "nnet_fwd.h"
#include "utility.h"

#include "matrix_multiply.h"

// Multiply matrices a and b with given sizes and store into result_goes_here.
//
// We could do something tricky by switching the role of result and temp, to
// avoid copying but let's leave that for now.
//
// result_temp is used to ensure that weird things don't happen if
// result_goes_here overlaps with a or b.
void matrix_multiply(float* a,
                     float* b,
                     int a_height,
                     int a_width_b_height,
                     int b_width,
                     float* result_goes_here,
                     float* result_temp) {

    int i, j, k;
    float value;

    // Initialize to zero
    int size = a_height * b_width;
    clear_matrix(result_temp, size);

matmul0:
    for (i = 0; i < a_height; i++) {
    matmul1:
        for (j = 0; j < b_width; j++) {
        matmul2:
            for (k = 0; k < a_width_b_height; k++) {
                value = conv_float2fixed(a[sub2ind(i, k, a_width_b_height)]) *
                        conv_float2fixed(b[sub2ind(k, j, b_width)]);
                result_temp[sub2ind(i, j, b_width)] =
                        conv_float2fixed(result_temp[sub2ind(i, j, b_width)] +
                                         conv_float2fixed(value));
            }
        }
    }
    copy_matrix(result_temp, result_goes_here, size);
}

// Multiply matrices a and b, assuming the last row of b are biases.
//
// Args:
//   a_height = height of A matrix.
//   b_height = height of the B matrix, which is also the width of the A matrix
//     + 1.
//   b_width = width of the B matrix.
void matrix_multiply_with_bias(float* a,
                               float* b,
                               int a_height,
                               int b_height,
                               int b_width,
                               float* result) {

    // a is hid, b is weights
    int i, j, k;
    float partial_sum;
    float value;

    int a_width = b_height - 1;

matmulb0:
    for (i = 0; i < a_height; i++) {
    matmulb1:
        for (j = 0; j < b_width; j++) {
            // Initialize to zero
            partial_sum = 0;
        matmulb2:
            for (k = 0; k < a_width; k++) {
                value = conv_float2fixed(a[sub2ind(i, k, a_width)]) *
                        conv_float2fixed(b[sub2ind(k, j, b_width)]);
                partial_sum += value;
            }
            // Add the bias (the index of the last row is the width of A).
            partial_sum += conv_float2fixed(b[sub2ind(a_width, j, b_width)]);
            result[sub2ind(i, j, b_width)] = partial_sum;
        }
    }
}

void matrix_multiply_with_bias_and_copy(float* a,
                                        float* b,
                                        int a_height,
                                        int b_height,
                                        int b_width,
                                        float* result_goes_here,
                                        float* result_temp) {
    int size = a_height * b_width;
    matrix_multiply_with_bias(
            a, b, a_height, b_height, b_width, result_temp);
    copy_matrix(result_temp, result_goes_here, size);
}

// Multiply the matrices a and b, but assume that b has been transposed.
//
// Args:
//   a_height = height of the A matrix.
//   b_height = height of the UNTRANSPOSED B matrix.
//   b_width = width of the UNTRANSPOSED B matrix.
void matrix_multiply_with_bias_transpose(float* a,
                                         float* b,
                                         int a_height,
                                         int b_height,
                                         int b_width,
                                         float* result) {

    // a is hid, b is weights
    int i, j, k;
    float partial_sum;
    float value;

    int a_width = b_height - 1;

matmulbt0:
    for (i = 0; i < a_height; i++) {
    matmulbt1:
        for (j = 0; j < b_width; j++) {
            // Initialize to zero
            partial_sum = 0;
        matmulbt2:
            for (k = 0; k < a_width; k++) {
                value = conv_float2fixed(a[sub2ind(i, k, a_width)]) *
                        conv_float2fixed(b[sub2ind(j, k, b_height)]);
                partial_sum += value;
            }
            // Add the bias.
            partial_sum += conv_float2fixed(b[sub2ind(j, a_width, b_height)]);
            result[sub2ind(i, j, b_width)] = partial_sum;
        }
    }
}
