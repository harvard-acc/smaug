#include "utility/utility.h"
#include "nnet_fwd.h"

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
    float input, weight, product;

    int a_width = b_height - 1;

    ARRAY_2D(float, _a, a, a_width);
    ARRAY_2D(float, _b, b, b_width);
    ARRAY_2D(float, _result, result, b_width);

matmulb0:
    for (i = 0; i < a_height; i++) {
    matmulb1:
        for (j = 0; j < b_width; j++) {
            // Initialize to zero
            partial_sum = 0;
        matmulb2:
            for (k = 0; k < a_width; k++) {
                input = conv_float2fixed(_a[i][k]);
                weight = conv_float2fixed(_b[k][j]);
                product = input * weight;
                partial_sum += product;
            }
            // Add the bias (the index of the last row is the width of A).
            partial_sum += conv_float2fixed(_b[a_width][j]);
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

// Multiply the matrices a and b, but assume that b has been transposed.
//
// Args:
//   a_height = height of the A matrix.
//   b_height = height of the TRANSPOSED B matrix.
//   b_width = width of the TRANSPOSED B matrix.
void matrix_multiply_with_bias_transpose(float* a,
                                         float* b,
                                         int a_height,
                                         int b_width,
                                         int b_height,
                                         float* result) {

    // a is hid, b is weights
    int i, j, k;
    float partial_sum;
    float value;

    int a_width = b_width - 1;

    ARRAY_2D(float, _a, a, a_width);
    ARRAY_2D(float, _b, b, b_width);
    ARRAY_2D(float, _result, result, b_height);

matmulbt0:
    for (i = 0; i < a_height; i++) {
    matmulbt1:
        for (j = 0; j < b_height; j++) {
            // Initialize to zero
            partial_sum = 0;
        matmulbt2:
            for (k = 0; k < a_width; k++) {
                value = conv_float2fixed(_a[i][k]) *
                        conv_float2fixed(_b[j][k]);
                partial_sum += value;
            }
            // Add the bias.
            partial_sum += conv_float2fixed(_b[j][a_width]);
            _result[i][j] = partial_sum;
        }
    }
}

// A = activations
// B = weights
// B must NOT be transposed!
void matrix_multiply_with_bias_smiv(float* a,
                                    float* b,
                                    int a_height,
                                    int b_height,
                                    int b_width,
                                    float* result) {

    int wgt_row, wgt_col, wgt_b;
    int act_batch;
    const int BLOCK_SIZE = 8;
    const int MAX_BATCH = 8;
    // TODO: Flatten this for Aladdin.
    float partial_sums[MAX_BATCH][BLOCK_SIZE];
    float input, weight, product, bias;

    int a_width = b_height - 1;

    printf("A = %dx%d, B=%dx%d (without biases)\n", a_height, a_width, b_height - 1, b_width);

    for (wgt_col = 0; wgt_col < b_width; wgt_col+=BLOCK_SIZE) {
        // Load in the bias.
        for (act_batch = 0; act_batch < a_height; act_batch++) {
            for (wgt_b = 0; wgt_b < BLOCK_SIZE && wgt_col + wgt_b < b_width; wgt_b++) {
                bias = conv_float2fixed(
                        b[sub2ind(a_width, wgt_col + wgt_b, b_width)]);
                partial_sums[act_batch][wgt_b] = bias;
            }
            print_debug(&partial_sums[0][0], 1, BLOCK_SIZE, BLOCK_SIZE);
        }

        for (wgt_row = 0; wgt_row < a_width; wgt_row++) {
            for (act_batch = 0; act_batch < a_height; act_batch++) {
                // MACC datapath.
                // Flatten this inner loop.
                for (wgt_b = 0; wgt_b < BLOCK_SIZE && wgt_col + wgt_b < b_width; wgt_b++) {
                    printf("Activation index: %d, %d, weight index: %d, %d\n",
                           act_batch, wgt_row, wgt_row, wgt_col + wgt_b);
                    input = conv_float2fixed(
                            a[sub2ind(act_batch, wgt_row, a_width)]);
                    weight = conv_float2fixed(
                            b[sub2ind(wgt_row, wgt_col + wgt_b, b_width)]);
                    product = input * weight;
                    printf("%4.4f x %4.4f = %4.4f\n", input, weight, product);
                    partial_sums[act_batch][wgt_b] += product;
                }
            }
        }

        print_debug(&partial_sums[0][0], 1, BLOCK_SIZE, BLOCK_SIZE);
        // Store to scratchpad.
        for (act_batch = 0; act_batch < a_height; act_batch++) {
            for (wgt_b = 0; wgt_b < BLOCK_SIZE && wgt_col + wgt_b < b_width; wgt_b++) {
                result[sub2ind(act_batch, wgt_col + wgt_b, b_width)] =
                        partial_sums[act_batch][wgt_b];
            }
        }
    }
}
