#include <assert.h>

#include "core/ref/activation_functions.h"
#include "utility/utility.h"
#include "nnet_fwd.h"

#include "impls.h"

/* Shift a single shift register left by shamt. */
ALWAYS_INLINE
static void shift_reg_lshift(float shift_reg[SHIFT_REG_SIZE], unsigned shamt) {
    unsigned sr;
    shift_reg_lshift_stage1:
    for (sr = 0; sr < SHIFT_REG_SIZE; sr++) {
        if (sr + shamt < SHIFT_REG_SIZE) {
            shift_reg[sr] = shift_reg[sr + shamt];
        } else {
            shift_reg[sr] = 0;
        }
    }
}

/* Shift a two shift registers in parallel left by shamt. */
ALWAYS_INLINE
static void shift_regs_lshift(float shift_reg0[SHIFT_REG_SIZE],
                              float shift_reg1[SHIFT_REG_SIZE],
                              unsigned shamt) {
    unsigned sr;
    shift_regs_lshift_stage1:
    for (sr = 0; sr < SHIFT_REG_SIZE; sr++) {
        if (sr + shamt < SHIFT_REG_SIZE) {
            shift_reg0[sr] = shift_reg0[sr + shamt];
            shift_reg1[sr] = shift_reg1[sr + shamt];
        } else {
            shift_reg0[sr] = 0;
            shift_reg1[sr] = 0;
        }
    }
}

ALWAYS_INLINE
static void conv_macc_datapath_fxp(float weights_buffer[VECTOR_SIZE],
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
        PRINT_MSG_V("psums\n");
        PRINT_DEBUG_V(&psums_0[0], 1, VECTOR_SIZE, VECTOR_SIZE);
        PRINT_DEBUG_V(&psums_1[0], 1, VECTOR_SIZE, VECTOR_SIZE);

        shift_regs_lshift(pipe0_shift_reg, pipe1_shift_reg, dp_shamt);
        PRINT_MSG_V("\nshift regs\n");
        PRINT_DEBUG_V(&pipe0_shift_reg[0], 1, SHIFT_REG_SIZE, SHIFT_REG_SIZE);
        PRINT_DEBUG_V(&pipe1_shift_reg[0], 1, SHIFT_REG_SIZE, SHIFT_REG_SIZE);
    }
}

// ALWAYS_INLINE
static void merge_psums_fxp(float psums_0[VECTOR_SIZE],
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
    PRINT_MSG_V("merged psums\n");
    PRINT_DEBUG_V(&result[0], 1, VECTOR_SIZE, VECTOR_SIZE);
}

// Perform a 3D convolution with one kernel on an image, without reduction.
//
// Args:
//   a: 3D array, indexed as [channel][row][col].
//   kernels: A 3D kernel, indexed as [channel][row][col].
//   curr_layer: Layer (or partial layer) configuration.
//   start_chan: Start reading the input from this channel.
//   result: a 3D array indexed as [channel][row][col].
//
// Returns:
//   The unreduced 3D partial sums in result.
void convolution3d_smiv_1kernel_noreduce_fxp(float* a,
                                             float* kernels,
                                             layer_t curr_layer,
                                             int start_chan,
                                             float* result) {
    int in_row, in_col, in_chan, out_row, out_col, sr, kern_row;
    unsigned j;

    const int a_height = curr_layer.inputs.rows;
    const int a_width = curr_layer.inputs.cols;
    const int a_pad = curr_layer.inputs.align_pad;

    const int result_height = curr_layer.outputs.rows;
    const int result_width = curr_layer.outputs.cols;
    const int result_pad = curr_layer.outputs.align_pad;

    // Filter is k_rows x k_cols x k_height.
    const int k_rows = curr_layer.weights.rows;
    const int k_cols = curr_layer.weights.cols;
    const int k_pad = curr_layer.weights.align_pad;

    // Convolution control parameters.
    // TODO: Refactor this into a scheduling pass.
    const int row_stride = curr_layer.stride.rows;
    const int in_col_stride = VECTOR_SIZE;
    const int k_col_stride = curr_layer.stride.cols;
    const int chan_stride = 1;
    const bool double_tp = k_cols < DATAPATH_WIDTH;
    const unsigned init_shamt = double_tp ? k_col_stride : DATAPATH_WIDTH;
    const unsigned dp_shamt = double_tp ? k_col_stride * 2 : k_col_stride;
    const unsigned input_fetches_per_row = FRAC_CEIL(a_width, VECTOR_SIZE);
    const unsigned last_input_pixel_start_col = result_width * k_col_stride;
    const bool has_boundary_case = last_input_pixel_start_col >
                             (input_fetches_per_row - 1) * VECTOR_SIZE;

    // Calculate max number of psums produced per VECTOR_SIZE activations per
    // datapath.
    unsigned max_psums_per_act;
    if (k_col_stride == 1)
        max_psums_per_act = double_tp ? DATAPATH_WIDTH : DATAPATH_WIDTH * 2;
    else if (k_col_stride == 2)
        max_psums_per_act = double_tp ? DATAPATH_WIDTH / 2: DATAPATH_WIDTH;
    else if (k_col_stride == 4)
        max_psums_per_act = DATAPATH_WIDTH / 2;
    else
        max_psums_per_act = 0;

    const int end_row = a_height - k_rows + 1;
    const int end_col = (has_boundary_case ? input_fetches_per_row
                                           : input_fetches_per_row - 1) *
                        VECTOR_SIZE;
    const int end_kern = k_rows;
    const int end_chan = curr_layer.inputs.height;

    ARRAY_3D(float, _a, a, a_height, a_width + a_pad);
    ARRAY_3D(float, _kernels, kernels, k_rows, k_cols + k_pad);
    ARRAY_3D(float, _result, result, result_height, result_width + result_pad);

    int end_col_marker = (input_fetches_per_row - 1) * VECTOR_SIZE;

    conv2d_chan:
    for (in_chan = 0; in_chan < end_chan; in_chan += chan_stride) {
        PRINT_MSG_V("Input channel %d\n", in_chan);
        out_row = 0;
        conv2d_row:
        for (in_row = 0; in_row < end_row; in_row += row_stride) {
            out_col = 0;
            conv2d_col:
            for (in_col = 0; in_col < end_col; in_col += in_col_stride) {
                // Compute schedule.
                // TODO: Refactor this...
                unsigned remaining_cols = result_width - out_col;
                unsigned remaining_per_dp, remainder, dp0_iters, dp1_iters, total_outpx;
                if (double_tp) {
                  remaining_per_dp = remaining_cols / 2;
                  remainder = remaining_cols % 2;
                  dp0_iters = min2(max_psums_per_act, remaining_per_dp + remainder);
                  dp1_iters = min2(max_psums_per_act, remaining_per_dp);
                  total_outpx = dp0_iters + dp1_iters;
                } else {
                  remaining_per_dp = remaining_cols;
                  dp0_iters = min2(max_psums_per_act, remaining_per_dp);
                  dp1_iters = min2(max_psums_per_act, remaining_per_dp);
                  total_outpx = dp0_iters;
                }
                PRINT_MSG_V("dp0_iters: %d, dp1_iters: %d\n", dp0_iters, dp1_iters);

                // Two partial sum regs, one for each pipe.
                float psums_0[VECTOR_SIZE] = { 0, 0, 0, 0, 0, 0, 0, 0 };
                float psums_1[VECTOR_SIZE] = { 0, 0, 0, 0, 0, 0, 0, 0 };
                conv2d_kern_row:
                for (kern_row = 0; kern_row < end_kern; kern_row ++) {
                    float weights_buffer[VECTOR_SIZE] = { 0, 0, 0, 0, 0, 0, 0, 0 };
                    float pipe0_shift_reg[SHIFT_REG_SIZE] = {
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                    };
                    float pipe1_shift_reg[SHIFT_REG_SIZE] = {
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                    };

                    // Load activations into shift registers.
                    conv2d_load_sr_pipe0:
                    for (sr = 0; sr < min2(VECTOR_SIZE, a_width - in_col); sr++) {
                        pipe0_shift_reg[sr] =
                                _a[in_chan + start_chan][in_row + kern_row]
                                  [in_col + sr];
                        pipe1_shift_reg[sr] =
                                _a[in_chan + start_chan][in_row + kern_row]
                                  [in_col + sr];
                    }
                    if (!(has_boundary_case && in_col == end_col_marker)) {
                        conv2d_load_sr_pipe1:
                        for (sr = 8; sr < min2(SHIFT_REG_SIZE, a_width - in_col);
                             sr++) {
                            pipe0_shift_reg[sr] =
                                    _a[in_chan + start_chan][in_row + kern_row]
                                      [in_col + sr];
                            pipe1_shift_reg[sr] =
                                    _a[in_chan + start_chan][in_row + kern_row]
                                      [in_col + sr];
                        }
                    }

                    PRINT_MSG_V("Shift registers after loading activations\n");
                    PRINT_DEBUG_V(&pipe0_shift_reg[0], 1, SHIFT_REG_SIZE, SHIFT_REG_SIZE);
                    PRINT_DEBUG_V(&pipe1_shift_reg[0], 1, SHIFT_REG_SIZE, SHIFT_REG_SIZE);

                    // Load weights into weights buffer, accounting for double tp
                    // mode.
                    if (double_tp) {
                      conv2d_load_wgts_double_tp:
                      for (int w = 0; w < k_cols; w++) {
                          float weight = _kernels[in_chan][kern_row][w];
                          weights_buffer[w] = weight;
                          weights_buffer[DATAPATH_WIDTH + w] = weight;
                      }
                    } else {
                      int bound = min2(k_cols, VECTOR_SIZE);
                      conv2d_load_wgts_single_tp:
                      for (int w = 0; w < bound; w++) {
                          weights_buffer[w] = _kernels[in_chan][kern_row][w];
                      }
                    }

                    PRINT_MSG_V("Weights buffer\n");
                    PRINT_DEBUG_V(&weights_buffer[0], 1, VECTOR_SIZE, VECTOR_SIZE);

                    shift_reg_lshift(pipe1_shift_reg, init_shamt);
                    PRINT_MSG_V("After initial shift of pipe1\n");
                    PRINT_DEBUG_V(&pipe1_shift_reg[0], 1, SHIFT_REG_SIZE, SHIFT_REG_SIZE);

                    // Primary datapath.
                    conv_macc_datapath_fxp(weights_buffer,
                                           pipe0_shift_reg,
                                           pipe1_shift_reg,
                                           dp_shamt,
                                           dp0_iters,
                                           dp1_iters,
                                           psums_0,
                                           psums_1);

                }

                float final_psums[VECTOR_SIZE] = { 0, 0, 0, 0, 0, 0, 0, 0 };
                merge_psums_fxp(psums_0, psums_1, double_tp, final_psums);

                // Aladdin does not allow us to directly pass an address to be
                // dereferenced as a temporary because all loads/stores must
                // originate from a named variable. A simple workaround is to just
                // explicitly cast here and use the casted variable's name.
                float* actfunc_temp = (float*)final_psums;
                activation_fun_fxp(actfunc_temp,
                                   1,
                                   VECTOR_SIZE,
                                   result_pad,
                                   curr_layer.activation);

                // This is the unreduced data!
                conv2d_commit:
                for (j = 0; j < total_outpx; j++)
                    _result[in_chan][out_row][out_col + j] = final_psums[j];
                out_col += total_outpx;
                if (out_col >= result_width)
                    out_col = 0;
            }
            PRINT_MSG_V("\nResult of row %d\n", out_row);
            PRINT_DEBUG_V(&_result[in_chan][out_row][0], 1, result_width, result_width);
            out_row++;
        }
    }
}
