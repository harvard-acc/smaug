#include <assert.h>

#include "core/activation_functions.h"
#include "utility/utility.h"
#include "nnet_fwd.h"

#include "smiv_core.h"

/* Shift a single shift register left by shamt. */
static void shift_reg_lshift(float shift_reg[SHIFT_REG_SIZE], unsigned shamt) {
    unsigned sr;
    float temp_shift_reg[SHIFT_REG_SIZE];
    shift_reg_lshift_stage1:
    for (sr = 0; sr < SHIFT_REG_SIZE; sr++) {
        if (sr + shamt < SHIFT_REG_SIZE) {
            temp_shift_reg[sr] = shift_reg[sr + shamt];
        } else {
            temp_shift_reg[sr] = 0;
        }
    }
    shift_reg_lshift_stage2:
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
    shift_regs_lshift_stage1:
    for (sr = 0; sr < SHIFT_REG_SIZE; sr++) {
        if (sr + shamt < SHIFT_REG_SIZE) {
            temp_shift_reg0[sr] = shift_reg0[sr + shamt];
            temp_shift_reg1[sr] = shift_reg1[sr + shamt];
        } else {
            temp_shift_reg0[sr] = 0;
            temp_shift_reg1[sr] = 0;
        }
    }
    shift_regs_lshift_stage2:
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

    conv2d_dp_outer:
    for (psum_reg = 0; psum_reg < dp0_iters; psum_reg++) {
        float accum_result_0 = psums_0[psum_reg];
        float accum_result_1 = psums_1[psum_reg];
        conv2d_dp_core:
        for (j = 0; j < DATAPATH_WIDTH; j++) {
            accum_result_0 += weights_buffer[j] * pipe0_shift_reg[j];
            accum_result_1 +=
                    weights_buffer[j + DATAPATH_WIDTH] * pipe1_shift_reg[j];
        }
        psums_0[psum_reg] = accum_result_0;
        // We have to shift the shift regs together in a single function call.
        if (psum_reg < dp1_iters)
          psums_1[psum_reg] = accum_result_1;
        PRINT_MSG("psums\n");
        PRINT_DEBUG(&psums_0[0], 1, VECTOR_SIZE, VECTOR_SIZE);
        PRINT_DEBUG(&psums_1[0], 1, VECTOR_SIZE, VECTOR_SIZE);

        shift_regs_lshift(pipe0_shift_reg, pipe1_shift_reg, dp_shamt);
        PRINT_MSG("\nshift regs\n");
        PRINT_DEBUG(&pipe0_shift_reg[0], 1, SHIFT_REG_SIZE, SHIFT_REG_SIZE);
        PRINT_DEBUG(&pipe1_shift_reg[0], 1, SHIFT_REG_SIZE, SHIFT_REG_SIZE);
    }
}

static void merge_psums(float psums_0[VECTOR_SIZE],
                        float psums_1[VECTOR_SIZE],
                        bool double_tp,
                        float result[VECTOR_SIZE]) {

    int i;

    if (double_tp) {
        merge_psums_double_tp:
        for (i = 0; i < VECTOR_SIZE/2; i ++) {
            result[2 * i] += psums_0[i];
            result[2 * i + 1] += psums_1[i];
        }
    } else {
        merge_psums_single_tp:
        for (i = 0; i < VECTOR_SIZE; i++) {
            result[i] += psums_0[i] + psums_1[i];
        }
    }
    PRINT_MSG("merged psums\n");
    PRINT_DEBUG(&result[0], 1, VECTOR_SIZE, VECTOR_SIZE);
}

// Perform a 2D convolution with one kernel and one input channel of one image.
//
// Args:
//   a: 4D array, indexed as [img][channel][row][col].
//   kernels: A stack of 3D kernels, indexed as [input_kern][channel][row][col].
//   img: Which input image this function is working on.
//   chan: Which channel of the input image.
//   curr_layer: Layer configuration.
//   result: a 3D array indexed as [input_chan][row][col].
//
// Returns:
//   The 2D convolution in result[chan].
static void convolution2d_smiv_1kernel_1channel(float* a,
                                                float* kernels,
                                                int img,
                                                int kern,
                                                int chan,
                                                layer_t curr_layer,
                                                float* result) {
    int in_row, in_col, out_row, out_col, sr, kern_row, j;

    const int a_height = curr_layer.input_rows;
    const int a_width = curr_layer.input_cols;
    const int a_pad = curr_layer.input_data_align_pad;

    const int result_height = curr_layer.output_rows;
    const int result_width = curr_layer.output_cols;
    const int result_pad = curr_layer.output_data_align_pad;

    // Filter is k_width x k_width x k_height.
    const int k_width = curr_layer.field_size;
    const int k_height =  curr_layer.input_height;
    const int k_stride = curr_layer.field_stride;

    // Convolution control parameters.
    // TODO: Refactor this into a scheduling pass.
    const int row_stride = k_stride;
    const int col_stride = VECTOR_SIZE;
    const bool double_tp = k_width < DATAPATH_WIDTH;
    const unsigned init_shamt = double_tp ? k_stride : DATAPATH_WIDTH;
    const unsigned dp_shamt = double_tp ? k_stride * 2 : k_stride;
    const unsigned input_fetches_per_row = FRAC_CEIL(a_width, VECTOR_SIZE);
    const unsigned last_input_pixel_start_col = result_width * k_stride;
    const bool has_boundary_case = last_input_pixel_start_col >
                             (input_fetches_per_row - 1) * VECTOR_SIZE;

    // Calculate max number of psums produced per VECTOR_SIZE activations per
    // datapath.
    unsigned max_psums_per_act;
    if (k_stride == 1)
        max_psums_per_act = double_tp ? DATAPATH_WIDTH : DATAPATH_WIDTH * 2;
    else if (k_stride == 2)
        max_psums_per_act = double_tp ? DATAPATH_WIDTH / 2: DATAPATH_WIDTH;
    else if (k_stride == 4)
        max_psums_per_act = DATAPATH_WIDTH / 2;
    else
        max_psums_per_act = 0;

    const int end_row = a_height - k_width + 1;
    const int end_col = (has_boundary_case ? input_fetches_per_row
                                           : input_fetches_per_row - 1) *
                        VECTOR_SIZE;
    const int end_kern = k_width;

    ARRAY_4D(float, _a, a, k_height, a_height, a_width + a_pad);
    ARRAY_4D(float, _kernels, kernels, k_height, k_width, k_width);
    ARRAY_3D(float, _result, result, result_height, result_width + result_pad);

    int end_col_marker = (input_fetches_per_row - 1) * 8;

    out_row = 0;
    conv2d_row:
    for (in_row = 0; in_row < end_row; in_row += row_stride) {
        out_col = 0;
        conv2d_col:
        for (in_col = 0; in_col < end_col; in_col += col_stride) {
            // Compute schedule.
            // TODO: Refactor this...
            unsigned remaining_cols = result_width - out_col;
            unsigned remaining_per_dp, remainder, dp0_iters, dp1_iters, total_outpx;
            if (double_tp) {
              remaining_per_dp = remaining_cols / 2;
              remainder = remaining_cols % 2;
              dp0_iters = min(max_psums_per_act, remaining_per_dp + remainder);
              dp1_iters = min(max_psums_per_act, remaining_per_dp);
              total_outpx = dp0_iters + dp1_iters;
            } else {
              remaining_per_dp = remaining_cols;
              dp0_iters = min(max_psums_per_act, remaining_per_dp);
              dp1_iters = min(max_psums_per_act, remaining_per_dp);
              total_outpx = dp0_iters;
            }
            PRINT_MSG("dp0_iters: %d, dp1_iters: %d\n", dp0_iters, dp1_iters);

            float final_psums[VECTOR_SIZE] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            conv2d_kern_row:
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
                conv2d_load_sr_pipe0:
                for (sr = 0; sr < min(VECTOR_SIZE, a_width - in_col); sr++) {
                    pipe0_shift_reg[sr] =
                            _a[img][chan][in_row + kern_row][in_col + sr];
                    pipe1_shift_reg[sr] =
                            _a[img][chan][in_row + kern_row][in_col + sr];
                }
                if (!(has_boundary_case && in_col == end_col_marker)) {
                    conv2d_load_sr_pipe1:
                    for (sr = 8; sr < min(SHIFT_REG_SIZE, a_width - in_col);
                         sr++) {
                        pipe0_shift_reg[sr] =
                                _a[img][chan][in_row + kern_row][in_col + sr];
                        pipe1_shift_reg[sr] =
                                _a[img][chan][in_row + kern_row][in_col + sr];
                    }
                }

                PRINT_MSG("Shift registers after loading activations\n");
                PRINT_DEBUG(&pipe0_shift_reg[0], 1, SHIFT_REG_SIZE, SHIFT_REG_SIZE);
                PRINT_DEBUG(&pipe1_shift_reg[0], 1, SHIFT_REG_SIZE, SHIFT_REG_SIZE);

                // Load weights into weights buffer, accounting for double tp
                // mode.
                if (double_tp) {
                  conv2d_load_wgts_double_tp:
                  for (int w = 0; w < k_width; w++) {
                      float weight = _kernels[kern][chan][kern_row][w];
                      weights_buffer[w] = weight;
                      weights_buffer[DATAPATH_WIDTH + w] = weight;
                  }
                } else {
                  int bound = min(k_width, VECTOR_SIZE);
                  conv2d_load_wgts_single_tp:
                  for (int w = 0; w < bound; w++) {
                      weights_buffer[w] = _kernels[kern][chan][kern_row][w];;
                  }
                }

                PRINT_MSG("Weights buffer\n");
                PRINT_DEBUG(&weights_buffer[0], 1, VECTOR_SIZE, VECTOR_SIZE);

                shift_reg_lshift(pipe1_shift_reg, init_shamt);
                PRINT_MSG("After initial shift of pipe1\n");
                PRINT_DEBUG(&pipe1_shift_reg[0], 1, SHIFT_REG_SIZE, SHIFT_REG_SIZE);

                // Primary datapath.
                conv_macc_datapath(weights_buffer,
                                   pipe0_shift_reg,
                                   pipe1_shift_reg,
                                   dp_shamt,
                                   dp0_iters,
                                   dp1_iters,
                                   psums_0,
                                   psums_1);

                merge_psums(psums_0, psums_1, double_tp, final_psums);
            }

            // This is the unreduced data!
            conv2d_commit:
            for (j = 0; j < total_outpx; j++)
                _result[chan][out_row][out_col + j] = final_psums[j];
            out_col += total_outpx;
            if (out_col >= result_width)
                out_col = 0;
        }
        PRINT_MSG("\nResult of row %d\n", out_row);
        PRINT_DEBUG(&_result[chan][out_row][0], 1, result_width, result_width);
        out_row++;
    }
}

void reduction_smiv(float *a,
                    layer_t curr_layer,
                    int img,
                    int kern,
                    float *result) {
    unsigned row, col, chan, c;

    const int result_height = curr_layer.output_rows;
    const int result_width = curr_layer.output_cols;
    const int result_pad = curr_layer.output_data_align_pad;
    const int padded_width = result_width + result_pad;

    const int k_height =  curr_layer.input_height;
    const int num_kerns = curr_layer.output_height;
    const bool run_activation = curr_layer.activation != NONE;

#ifdef TRACE_MODE
    assert(padded_width % VECTOR_SIZE &&
           "Padded width must be multiple of VECTOR_SIZE!");
#endif

    ARRAY_3D(float, _a, a, result_height, padded_width);
    ARRAY_4D(float, _result, result, num_kerns, result_height, padded_width);

    reduction_row:
    for (row = 0; row < result_height; row++) {
        reduction_col:
        for (col = 0; col < padded_width; col += VECTOR_SIZE) {
            float partial_sums[VECTOR_SIZE] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            reduction_chan:
            for (chan = 0; chan < k_height; chan++) {
                // TODO: Remove the additional col + c check.
                reduction_core:
                for (c = 0; (c < VECTOR_SIZE) && (col + c < result_width); c++) {
                    partial_sums[c] += _a[chan][row][col + c];
                }
            }

            if (run_activation) {
                activation_fun(&partial_sums[0], VECTOR_SIZE, RELU, NULL);
            }

            reduction_commit:
            for (c = 0; (c < VECTOR_SIZE) && (col + c < result_width); c++) {
                _result[img][kern][row][col + c] = partial_sums[c];
            }
        }
    }
}

void convolution2d_smiv(float* a,
                        float* kernels,
                        layer_t curr_layer,
                        float* result) {
    int ni, nk, nc;

    const int input_height = curr_layer.input_height;
    const int input_rows = curr_layer.input_rows;
    const int input_cols = curr_layer.input_cols;
    const int input_pad = curr_layer.input_data_align_pad;
    const int num_kerns = curr_layer.output_height;

    // Stores the unreduced convolution output.
    // TODO: This may become an issue with the stack size.
    float temp[input_height][input_rows][input_cols + input_pad];

    // PRINT_DEBUG4D(a, input_rows, input_cols + input_pad, input_height);

    conv2d_per_image:
    for (ni = 0; ni < NUM_TEST_CASES; ni++) {
        // Loop over all inputs in this batch.
        conv2d_per_kernel:
        for (nk = 0; nk < num_kerns; nk++) {
            conv2d_per_chan:
            for (nc = 0; nc < input_height; nc++) {
                convolution2d_smiv_1kernel_1channel(
                        a, kernels, ni, nk, nc, curr_layer, &temp[0][0][0]);
            }
            reduction_smiv(&temp[0][0][0], curr_layer, ni, nk, result);
        }
    }
}

// A = activations
// B = weights
// B must NOT be transposed!
//
// This works on a batch of inputs, reusing the model weights for each input activation.
void matrix_multiply_with_bias_smiv_batch_fxp(float* a,
                                              float* b,
                                              int a_height,
                                              int b_height,
                                              int b_width,
                                              bool run_activation,
                                              float* result) {
    int wgt_row, wgt_col, wgt_b;
    int act_batch;
    float partial_sums[MAX_BATCH][VECTOR_SIZE];
    float input, weight, product, bias;

    int a_width = b_height - 1;

    ARRAY_2D(float, _a, a, a_width);
    ARRAY_2D(float, _b, b, b_width);
    ARRAY_2D(float, _result, result, b_width);

#ifndef TRACE_MODE
    assert(b_width % VECTOR_SIZE == 0 &&
           "Width of weights must be a multiple of VECTOR_SIZE!");
#endif

    wgt_col:
    for (wgt_col = 0; wgt_col < b_width; wgt_col+=VECTOR_SIZE) {
        // Load in the bias.
        load_bias_batch:
        for (act_batch = 0; act_batch < a_height; act_batch++) {
            load_bias:
            for (wgt_b = 0; wgt_b < VECTOR_SIZE; wgt_b++) {
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
                for (wgt_b = 0; wgt_b < VECTOR_SIZE; wgt_b++) {
                    input = conv_float2fixed(_a[act_batch][wgt_row]);
                    weight = conv_float2fixed(_b[wgt_row][wgt_col + wgt_b]);

                    product = input * weight;
                    partial_sums[act_batch][wgt_b] += product;
                }
            }
        }

        // Run through activation function.
        if (run_activation) {
            run_activation_func:
            for (act_batch = 0; act_batch < a_height; act_batch++)
                activation_fun(&partial_sums[act_batch][0], VECTOR_SIZE, RELU, NULL);
        }

        // Store to scratchpad.
        act_batch_store:
        for (act_batch = 0; act_batch < a_height; act_batch++) {
            wgt_b_store:
            for (wgt_b = 0; wgt_b < VECTOR_SIZE; wgt_b++) {
                _result[act_batch][wgt_col + wgt_b] = partial_sums[act_batch][wgt_b];
            }
        }
    }
}

// A = activations
// B = weights
// B must NOT be transposed!
//
// This works on a single input at a time, so it does not store partial sums
// for more than one input vector and does not exploit weight reuse.
//
// This implementation lets Aladdin do a better job of imitating performance
// when working on one input (to avoid a useless layer of inner looping that
// introduces additional overheads).
void matrix_multiply_with_bias_smiv_nobatch_fxp(float* a,
                                                float* b,
                                                int a_height,
                                                int b_height,
                                                int b_width,
                                                bool run_activation,
                                                float* result) {
    int wgt_row, wgt_col, wgt_b, input_act;
    float partial_sums[VECTOR_SIZE];
    float input, weight, product, bias;

    int a_width = b_height - 1;

    ARRAY_2D(float, _a, a, a_width);
    ARRAY_2D(float, _b, b, b_width);
    ARRAY_2D(float, _result, result, b_width);

#ifndef TRACE_MODE
    assert(b_width % VECTOR_SIZE == 0 &&
           "Width of weights must be a multiple of VECTOR_SIZE!");
#endif

    input_act:
    for (input_act = 0; input_act < a_height; input_act++) {
        wgt_col:
        for (wgt_col = 0; wgt_col < b_width; wgt_col+=VECTOR_SIZE) {
            // Load in the bias.
            load_bias:
            for (wgt_b = 0; wgt_b < VECTOR_SIZE; wgt_b++) {
                bias = conv_float2fixed(_b[a_width][wgt_col + wgt_b]);
                partial_sums[wgt_b] = bias;
            }

            wgt_row:
            for (wgt_row = 0; wgt_row < a_width; wgt_row++) {
                // MACC datapath.
                // Flatten this inner loop.
                wgt_b_macc:
                for (wgt_b = 0; wgt_b < VECTOR_SIZE; wgt_b++) {
                    input = conv_float2fixed(_a[input_act][wgt_row]);
                    weight = conv_float2fixed(_b[wgt_row][wgt_col + wgt_b]);

                    product = input * weight;
                    partial_sums[wgt_b] += product;
                }
            }

            // Run through activation function.
            if (run_activation) {
                activation_fun(&partial_sums[0], VECTOR_SIZE, RELU, NULL);
            }

            // Store to scratchpad.
            wgt_b_store:
            for (wgt_b = 0; wgt_b < VECTOR_SIZE; wgt_b++) {
                _result[input_act][wgt_col + wgt_b] = partial_sums[wgt_b];
            }
        }
    }
}

void matrix_multiply_with_bias_smiv(float* a,
                                    float* b,
                                    int a_height,
                                    int b_height,
                                    int b_width,
                                    bool run_activation,
                                    float* result) {
#ifdef DISABLE_SMIV_INPUT_BATCHING
    matrix_multiply_with_bias_smiv_nobatch_fxp(
            a, b, a_height, b_height, b_width, run_activation, result);
#else
    matrix_multiply_with_bias_smiv_batch_fxp(
            a, b, a_height, b_height, b_width, run_activation, result);
#endif
}
