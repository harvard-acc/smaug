#include "utility/utility.h"
#include "nnet_fwd.h"

#include "smiv_core.h"

const unsigned VECTOR_SIZE = 8;
const unsigned DATAPATH_WIDTH = 4;
const unsigned SHIFT_REG_SIZE = 16;
const unsigned MAX_BATCH = 8;

/* Shift a single shift register left by shamt. */
static void shift_reg_lshift(float shift_reg[SHIFT_REG_SIZE], unsigned shamt) {
    unsigned sr;
    float temp_shift_reg[SHIFT_REG_SIZE];
    for (sr = 0; sr < SHIFT_REG_SIZE; sr++) {
        if (sr + shamt < SHIFT_REG_SIZE) {
            temp_shift_reg[sr] = shift_reg[sr + shamt];
        } else {
            temp_shift_reg[sr] = 0;
        }
    }
    for (sr = 0; sr < SHIFT_REG_SIZE; sr++) {
        shift_reg[sr] = temp_shift_reg[sr];
    }
}

/* Shift a two shift registers in parallel left by shamt. */
static void shift_regs_lshift(float shift_reg0[SHIFT_REG_SIZE],
                              float shift_reg1[SHIFT_REG_SIZE],
                              unsigned shamt) {
    unsigned sr;
    float temp_shift_reg0[SHIFT_REG_SIZE];
    float temp_shift_reg1[SHIFT_REG_SIZE];
    for (sr = 0; sr < SHIFT_REG_SIZE; sr++) {
        if (sr + shamt < SHIFT_REG_SIZE) {
            temp_shift_reg0[sr] = shift_reg0[sr + shamt];
            temp_shift_reg1[sr] = shift_reg1[sr + shamt];
        } else {
            temp_shift_reg0[sr] = 0;
            temp_shift_reg1[sr] = 0;
        }
    }
    for (sr = 0; sr < SHIFT_REG_SIZE; sr++) {
        shift_reg0[sr] = temp_shift_reg0[sr];
        shift_reg1[sr] = temp_shift_reg1[sr];
    }
}

static void conv_macc_datapath(float weights_buffer[VECTOR_SIZE],
                               float pipe0_shift_reg[SHIFT_REG_SIZE],
                               float pipe1_shift_reg[SHIFT_REG_SIZE],
                               unsigned k_stride,
                               float partial_sums[2][VECTOR_SIZE]) {
    unsigned psum_reg, j;

    for (psum_reg = 0; psum_reg < VECTOR_SIZE; psum_reg++) {
        // Do the MACCs for datapath 0.
        float accum_result_0 = partial_sums[0][psum_reg];
        for (j = 0; j < DATAPATH_WIDTH; j += k_stride) {
            accum_result_0 += weights_buffer[j] * pipe0_shift_reg[j];
        }

        // Now for datapath 1.
        float accum_result_1 = partial_sums[1][psum_reg];
        for (j = 0; j < DATAPATH_WIDTH; j += k_stride) {
            accum_result_1 += weights_buffer[j] * pipe1_shift_reg[j];
        }

        partial_sums[0][psum_reg] = accum_result_0;
        partial_sums[1][psum_reg] = accum_result_1;

        shift_regs_lshift(pipe0_shift_reg, pipe1_shift_reg, k_stride);
    }
}

static void convolution2d_kernel_smiv_1channel(float* a,
                                               float* kernels,
                                               int img,
                                               int kern,
                                               int chan,
                                               layer_t curr_layer,
                                               float* result) {
    int out_row, out_col, sr, kern_row, j;

    const int a_height = curr_layer.input_rows;
    const int a_width = curr_layer.input_cols;

    const int result_height = curr_layer.output_rows;
    const int result_width = curr_layer.output_cols;

    // Filter is k_width x k_width x k_height.
    const int k_width = curr_layer.field_size;
    const int k_height =  curr_layer.input_height;
    const int k_stride = curr_layer.field_stride;
    const int num_kerns = curr_layer.output_height;

    // Convolution borders.
    const int start_row = 0;
    const int start_col = 0;
    const int end_row = result_width;
    const unsigned init_shamt = VECTOR_SIZE;
    // const int end_col = result_height;

    const int row_stride = k_stride;
    const int col_stride = VECTOR_SIZE;

    ARRAY_4D(float, _a, a, k_height, a_height, a_width);
    ARRAY_4D(float, _kernels, kernels, k_height, k_width, k_width);
    ARRAY_4D(float, _result, result, num_kerns, result_height, result_width);

    float pipe0_shift_reg[SHIFT_REG_SIZE];
    float pipe1_shift_reg[SHIFT_REG_SIZE];
    float weights_buffer[VECTOR_SIZE];
    // Two partial sum regs, one for each pipe.
    float partial_sums[2][VECTOR_SIZE];

    for (out_row = start_row; out_row < end_row; out_row += row_stride) {
        for (out_col = start_col; out_col < VECTOR_SIZE;
             out_col += col_stride) {
            for (kern_row = 0; kern_row < k_height; kern_row += k_stride) {
                // Load activations into shift registers.
                for (sr = 0; sr < max(VECTOR_SIZE, a_width); sr++) {
                    pipe0_shift_reg[sr] =
                            _a[img][chan][out_row + kern_row][out_col + sr];
                    pipe1_shift_reg[sr] =
                            _a[img][chan][out_row + kern_row][out_col + sr];
                }
                for (sr = 8; sr < max(SHIFT_REG_SIZE, a_width); sr++) {
                    pipe0_shift_reg[sr] =
                            _a[img][chan][out_row + kern_row][out_col + sr];
                    pipe1_shift_reg[sr] =
                            _a[img][chan][out_row + kern_row][out_col + sr];
                }

                // Load weights into weights buffer.
                for (int w = 0; w < VECTOR_SIZE; w++) {
                    weights_buffer[w] = _kernels[kern][chan][kern_row][w];
                }

                // TODO: Initial shifts of shift registers...
                shift_reg_lshift(pipe0_shift_reg, init_shamt);

                // Primary datapath.
                conv_macc_datapath(weights_buffer,
                                   pipe0_shift_reg,
                                   pipe1_shift_reg,
                                   k_stride,
                                   partial_sums);

                // TODO: This is the unreduced data!
                for (j = 0; j < VECTOR_SIZE; j++)
                  _result[img][chan][out_row][out_col + j] = partial_sums[0][j];
            }
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
    float partial_sums[MAX_BATCH][VECTOR_SIZE];
    float input, weight, product, bias;

    int a_width = b_height - 1;

    ARRAY_2D(float, _a, a, a_width);
    ARRAY_2D(float, _b, b, b_width);
    ARRAY_2D(float, _result, result, b_width);

    wgt_col:
    for (wgt_col = 0; wgt_col < b_width; wgt_col+=VECTOR_SIZE) {
        // Load in the bias.
        load_bias:
        for (act_batch = 0; act_batch < a_height; act_batch++) {
            for (wgt_b = 0; wgt_b < VECTOR_SIZE && wgt_col + wgt_b < b_width; wgt_b++) {
                bias = conv_float2fixed(_b[a_width][wgt_col + wgt_b]);
                partial_sums[act_batch][wgt_b] = bias;
            }
        }

        wgt_row:
        for (wgt_row = 0; wgt_row < a_width; wgt_row++) {
            act_batch_macc:
            for (act_batch = 0; act_batch < a_height; act_batch++) {
                // MACC datapath.
                // Flatten this inner loop.
                wgt_b_macc:
                for (wgt_b = 0; wgt_b < VECTOR_SIZE && wgt_col + wgt_b < b_width; wgt_b++) {
                    input = conv_float2fixed(_a[act_batch][wgt_row]);
                    weight = conv_float2fixed(_b[wgt_row][wgt_col + wgt_b]);

                    product = input * weight;
                    partial_sums[act_batch][wgt_b] += product;
                }
            }
        }

        // Store to scratchpad.
        act_batch_store:
        for (act_batch = 0; act_batch < a_height; act_batch++) {
            wgt_b_store:
            for (wgt_b = 0; wgt_b < VECTOR_SIZE && wgt_col + wgt_b < b_width; wgt_b++) {
                _result[act_batch][wgt_col + wgt_b] = partial_sums[act_batch][wgt_b];
            }
        }
    }
}

void convolution2d_kernel_smiv(float* a,
                               float* kernels,
                               int img,
                               int kern,
                               layer_t curr_layer,
                               float* result) {
    int d;

    for (d = 0; d < curr_layer.input_height; d++) {
        convolution2d_kernel_smiv_1channel(
                a, kernels, img, kern, d, curr_layer, result);
    }
}
