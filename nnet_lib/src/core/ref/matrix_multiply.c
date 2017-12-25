#include "core/ref/matrix_multiply.h"
#include "utility/utility.h"
#include "nnet_fwd.h"

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
    ARRAY_2D(float, _a, a, a_width_b_height);
    ARRAY_2D(float, _b, b, b_width);
    ARRAY_2D(float, _result, result_temp, b_width);

    // Initialize to zero
    int size = a_height * b_width;
    clear_matrix(result_temp, size);

matmul0:
    for (i = 0; i < a_height; i++) {
    matmul1:
        for (j = 0; j < b_width; j++) {
        matmul2:
            for (k = 0; k < a_width_b_height; k++) {
                value = conv_float2fixed(_a[i][k]) *
                        conv_float2fixed(_b[k][j]);
                _result[i][j] = conv_float2fixed(_result[i][j]) +
                                conv_float2fixed(value);
            }
        }
    }
    copy_matrix(result_temp, result_goes_here, size);
}

// Multiply matrices a and b, assuming both are row major and the last row of b
// are biases.
//
// Args:
//   a_height = height of A matrix.
//   b_height = height of the B matrix, which is also the width of the A matrix
//     + 1.
//   b_width = width of the B matrix.
void matrix_multiply_with_bias(float* __restrict__ a,
                               float* __restrict__ b,
                               int a_height,
                               int b_height,
                               int b_width,
                               float* __restrict__ result) {

    // a is hid, b is weights
    int i, j, k;
    float partial_sum;
    float input, weight, product;

    int a_width = b_height - 1;

    ARRAY_2D(float, _a, a, a_width);
    ARRAY_2D(float, _b, b, b_width);
    ARRAY_2D(float, _result, result, b_width);

    PRINT_MSG_V("B matrix:\n");
    PRINT_DEBUG_V(b, b_height, b_width, b_width);

matmulb0:
    for (i = 0; i < a_height; i++) {
    matmulb1:
        for (j = 0; j < b_width; j++) {
            // Preload partial_sum with the bias (the index of the last row is
            // the width of A).
            partial_sum = conv_float2fixed(_b[a_width][j]);

        matmulb2:
            for (k = 0; k < a_width; k++) {
                input = conv_float2fixed(_a[i][k]);
                weight = conv_float2fixed(_b[k][j]);
                product = input * weight;
                partial_sum += product;
            }
            _result[i][j] = partial_sum;
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

// Multiply the matrices a and b, but assume that b has been transposed (col
// major).
//
// The biases are stored after all elements in b.
//
// Args:
//   a_height = height of the A matrix.
//   b_height = height of the TRANSPOSED B matrix.
//   b_width = width of the TRANSPOSED B matrix + 1.
void matrix_multiply_with_bias_transpose(float* __restrict__ a,
                                         float* __restrict__ b,
                                         int a_height,
                                         int b_width,
                                         int b_height,
                                         float* __restrict__ result) {

    // a is hid, b is weights
    int i, j, k;
    float partial_sum;
    float value;

    b_width--;  // b_width originally accounted for the biases, but the biases
                // are no longer part of the main array.
    int a_width = b_width;

    ARRAY_2D(float, _a, a, a_width);
    ARRAY_2D(float, _b, b, b_width);
    ARRAY_2D(float, _result, result, b_height);  // The width of the untransposed B.

    PRINT_MSG_V("B matrix transpose:\n");
    PRINT_DEBUG_V(b, b_height, b_width, b_width);

matmulbt0:
    for (i = 0; i < a_height; i++) {
    matmulbt1:
        for (j = 0; j < b_height; j++) {
            // Preload the bias.
            // This indexing trick will let us safely access "beyond" the b
            // matrix to get to the biases.
            partial_sum = conv_float2fixed(_b[b_height][j]);
        matmulbt2:
            for (k = 0; k < a_width; k++) {
                value = conv_float2fixed(_a[i][k]) *
                        conv_float2fixed(_b[j][k]);
                partial_sum += value;
            }
            _result[i][j] = partial_sum;
        }
    }
}
