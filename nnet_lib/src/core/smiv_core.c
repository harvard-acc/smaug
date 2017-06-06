#include "utility/utility.h"
#include "nnet_fwd.h"

#include "smiv_core.h"

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
                               unsigned dp_shamt,
                               unsigned dp0_iters,
                               unsigned dp1_iters,
                               float psums_0[VECTOR_SIZE],
                               float psums_1[VECTOR_SIZE]) {
    unsigned psum_reg, j;

    for (psum_reg = 0; psum_reg < dp0_iters; psum_reg++) {
        float accum_result_0 = psums_0[psum_reg];
        float accum_result_1 = psums_1[psum_reg];
        for (j = 0; j < DATAPATH_WIDTH; j++) {
            accum_result_0 += weights_buffer[j] * pipe0_shift_reg[j];
            accum_result_1 +=
                    weights_buffer[j + DATAPATH_WIDTH] * pipe1_shift_reg[j];
        }
        psums_0[psum_reg] = accum_result_0;
        // We have to shift the shift regs together in a single function call.
        if (psum_reg < dp1_iters)
          psums_1[psum_reg] = accum_result_1;
        shift_regs_lshift(pipe0_shift_reg, pipe1_shift_reg, dp_shamt);
    }
}

static void merge_psums(float psums_0[VECTOR_SIZE],
                        float psums_1[VECTOR_SIZE],
                        bool double_tp,
                        float result[VECTOR_SIZE]) {

    int i;

    // TODO: Should we consider num_valid_accum? It's a lot of complexity...
    if (double_tp) {
        for (i = 0; i < VECTOR_SIZE; i += 2) {
            result[i] = psums_0[i];
            result[i + 1] = psums_1[i];
        }
    } else {
        for (i = 0; i < VECTOR_SIZE; i++) {
            result[i] = psums_0[i] + psums_1[i];
        }
    }
}

static void convolution2d_smiv_1kernel_1channel(float* a,
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
    const int end_row = result_width;
    const int end_col = result_height;
    const int end_kern = k_height;

    // Convolution control parameters.
    // TODO: Refactor this into a scheduling pass.
    const int row_stride = k_stride;
    const int col_stride = VECTOR_SIZE;
    const bool double_tp = k_width < DATAPATH_WIDTH;
    const unsigned init_shamt = double_tp ? k_stride : DATAPATH_WIDTH;
    const unsigned dp_shamt = double_tp ? k_stride * 2 : k_stride;
    const unsigned max_psums_per_act =
            double_tp ? DATAPATH_WIDTH : DATAPATH_WIDTH * 2;

    ARRAY_4D(float, _a, a, k_height, a_height, a_width);
    ARRAY_4D(float, _kernels, kernels, k_height, k_width, k_width);
    ARRAY_4D(float, _result, result, num_kerns, result_height, result_width);

    for (out_row = 0; out_row < end_row; out_row += row_stride) {
        for (out_col = 0; out_col < end_col; out_col += col_stride) {
            // Compute schedule.
            // TODO: Refactor this...
            unsigned remaining_cols = result_width - out_col;
            unsigned remaining_per_dp = remaining_cols / 2;
            unsigned remainder = remaining_cols % 2;
            unsigned dp0_iters = min(max_psums_per_act, remaining_per_dp + remainder);
            unsigned dp1_iters = min(max_psums_per_act, remaining_per_dp);

            for (kern_row = 0; kern_row < end_kern; kern_row ++) {
                float weights_buffer[VECTOR_SIZE] = { 0, 0, 0, 0, 0, 0, 0, 0 };
                float pipe0_shift_reg[SHIFT_REG_SIZE] = {
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                };
                float pipe1_shift_reg[SHIFT_REG_SIZE] = {
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                };

                // Two partial sum regs, one for each pipe.
                float psums_0[VECTOR_SIZE] = { 0, 0, 0, 0, 0, 0, 0, 0 };
                float psums_1[VECTOR_SIZE] = { 0, 0, 0, 0, 0, 0, 0, 0 };

                // Load activations into shift registers.
                for (sr = 0; sr < max(VECTOR_SIZE, a_width - out_col); sr++) {
                    pipe0_shift_reg[sr] =
                            _a[img][chan][out_row + kern_row][out_col + sr];
                    pipe1_shift_reg[sr] =
                            _a[img][chan][out_row + kern_row][out_col + sr];
                }
                for (sr = 8; sr < max(SHIFT_REG_SIZE, a_width - out_col); sr++) {
                    pipe0_shift_reg[sr] =
                            _a[img][chan][out_row + kern_row][out_col + sr];
                    pipe1_shift_reg[sr] =
                            _a[img][chan][out_row + kern_row][out_col + sr];
                }

                // Load weights into weights buffer, accounting for double tp
                // mode.
                if (double_tp) {
                  for (int w = 0; w < DATAPATH_WIDTH; w++) {
                      float weight = _kernels[kern][chan][kern_row][w];
                      weights_buffer[w] = weight;
                      weights_buffer[DATAPATH_WIDTH + w] = weight;
                  }
                } else {
                  int bound = min(k_width, VECTOR_SIZE);
                  for (int w = 0; w < bound; w++) {
                      weights_buffer[w] = _kernels[kern][chan][kern_row][w];;
                  }
                }

                shift_reg_lshift(pipe1_shift_reg, init_shamt);

                // Primary datapath.
                conv_macc_datapath(weights_buffer,
                                   pipe0_shift_reg,
                                   pipe1_shift_reg,
                                   dp_shamt,
                                   dp0_iters,
                                   dp1_iters,
                                   psums_0,
                                   psums_1);

                float final_psums[VECTOR_SIZE] = { 0, 0, 0, 0, 0, 0, 0, 0 };
                merge_psums(psums_0, psums_1, double_tp, final_psums);

                // TODO: This is the unreduced data!
                // TODO: Is dp0_iters the correct bound?
                for (j = 0; j < dp0_iters; j++)
                  _result[img][chan][out_row][out_col + j] = final_psums[j];
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

void convolution2d_smiv(float* a,
                        float* kernels,
                        layer_t curr_layer,
                        float* result) {
    int ni, nk, nc;

    print_debug4d(a,
                  curr_layer.input_rows,
                  curr_layer.input_cols,
                  curr_layer.input_height);

conv2d_per_image:
    for (ni = 0; ni < NUM_TEST_CASES; ni++) {
        // Loop over all inputs in this batch.
    conv2d_per_kernel:
        for (nk = 0; nk < curr_layer.output_height; nk++) {
        conv2d_per_chan:
            for (nc = 0; nc < curr_layer.input_height; nc++) {
                convolution2d_smiv_1kernel_1channel(
                        a, kernels, ni, nk, nc, curr_layer, result);
            }
        }
    }
}
