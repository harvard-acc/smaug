#include <assert.h>

#include "core/smiv/activation_functions_simd.h"
#include "core/smiv/impls.h"
#include "utility/utility.h"
#include "nnet_fwd.h"

#define SR_SIZE ((SHIFT_REG_SIZE)/(VECTOR_SIZE))

/* Shift a single shift register left by shamt. */
ALWAYS_INLINE
static void shift_reg_simd_lshift(v8fp_t shift_reg[SR_SIZE], unsigned shamt) {
    unsigned sr, sr0, sr1, sr2, sr3, shidx;
    shift_reg_lshift:
    for (sr = 0; sr < SHIFT_REG_SIZE; sr++) {
        shidx = sr + shamt;
        // Destination indices.
        sr0 = sr / VECTOR_SIZE;
        sr1 = sr % VECTOR_SIZE;
        // Source indices.
        sr2 = shidx / VECTOR_SIZE;
        sr3 = shidx % VECTOR_SIZE;
        if (shidx < SHIFT_REG_SIZE) {
            shift_reg[sr0][sr1] = shift_reg[sr2][sr3];
        } else {
            shift_reg[sr0][sr1] = 0;
        }
    }
}

/* Shift a two shift registers in parallel left by shamt. */
ALWAYS_INLINE
static void shift_regs_simd_lshift(v8fp_t shift_reg0[SR_SIZE],
                                   v8fp_t shift_reg1[SR_SIZE],
                                   unsigned shamt) {
    unsigned sr, sr0, sr1, sr2, sr3, shidx;
    shift_regs_lshift:
    for (sr = 0; sr < SHIFT_REG_SIZE; sr++) {
        shidx = sr + shamt;
        // Destination indices.
        sr0 = sr / VECTOR_SIZE;
        sr1 = sr % VECTOR_SIZE;
        // Source indices.
        sr2 = shidx / VECTOR_SIZE;
        sr3 = shidx % VECTOR_SIZE;
        if (shidx < SHIFT_REG_SIZE) {
            shift_reg0[sr0][sr1] = shift_reg0[sr2][sr3];
            shift_reg1[sr0][sr1] = shift_reg1[sr2][sr3];
        } else {
            shift_reg0[sr0][sr1] = 0;
            shift_reg1[sr0][sr1] = 0;
        }
    }
}

ALWAYS_INLINE
static void conv_macc_datapath_simd_fxp(v8fp_t weights_buffer,
                                        v8fp_t pipe0_shift_reg[SR_SIZE],
                                        v8fp_t pipe1_shift_reg[SR_SIZE],
                                        unsigned dp_shamt,
                                        unsigned dp0_iters,
                                        unsigned dp1_iters,
                                        v8fp_t psums_0[1],
                                        v8fp_t psums_1[1]) {
    unsigned psum_reg, j;
    v8fp_t psums_tmp_0 = psums_0[0];
    v8fp_t psums_tmp_1 = psums_1[0];

    conv2d_dp_outer:
    for (psum_reg = 0; psum_reg < dp0_iters; psum_reg++) {
        float accum_result_0 = psums_tmp_0[psum_reg];
        float accum_result_1 = psums_tmp_1[psum_reg];
        conv2d_dp_core:
        for (j = 0; j < DATAPATH_WIDTH; j++) {
            accum_result_0 += weights_buffer[j] * pipe0_shift_reg[0][j];
            accum_result_1 +=
                    weights_buffer[j + DATAPATH_WIDTH] * pipe1_shift_reg[0][j];
        }
        psums_tmp_0[psum_reg] = accum_result_0;
        // We have to shift the shift regs together in a single function call.
        if (psum_reg < dp1_iters)
          psums_tmp_1[psum_reg] = accum_result_1;
        PRINT_MSG_V("psums\n");
        PRINT_DEBUG_V(&psums_tmp_0[0], 1, VECTOR_SIZE, VECTOR_SIZE);
        PRINT_DEBUG_V(&psums_tmp_1[0], 1, VECTOR_SIZE, VECTOR_SIZE);

        shift_regs_simd_lshift(pipe0_shift_reg, pipe1_shift_reg, dp_shamt);
        PRINT_MSG_V("\nshift regs\n");
        PRINT_DEBUG_V(&pipe0_shift_reg[0][0], 1, SHIFT_REG_SIZE, SHIFT_REG_SIZE);
        PRINT_DEBUG_V(&pipe1_shift_reg[0][0], 1, SHIFT_REG_SIZE, SHIFT_REG_SIZE);
    }

    psums_0[0] = psums_tmp_0;
    psums_1[0] = psums_tmp_1;
}

ALWAYS_INLINE
static v8fp_t merge_psums_simd_fxp(v8fp_t psums_0,
                                   v8fp_t psums_1,
                                   bool double_tp) {

    int i;

    v8fp_t result_tmp = (v8fp_t){ 0, 0, 0, 0, 0, 0, 0, 0 };

    if (double_tp) {
        merge_psums_double_tp:
        for (i = 0; i < VECTOR_SIZE/2; i ++) {
            result_tmp[2 * i] += psums_0[i];
            result_tmp[2 * i + 1] += psums_1[i];
        }
    } else {
        merge_psums_single_tp:
        for (i = 0; i < VECTOR_SIZE; i++) {
            result_tmp[i] += psums_0[i] + psums_1[i];
        }
    }
    PRINT_MSG_V("merged psums\n");
    PRINT_DEBUG_V(&result_tmp[0], 1, VECTOR_SIZE, VECTOR_SIZE);
    return result_tmp;
}

// Perform a 3D convolution with one kernel on an image, without reduction.
//
// This implemention uses SIMD extensions.
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
void convolution3d_smiv_1kernel_noreduce_simd_fxp(float* a,
                                                  float* kernels,
                                                  layer_t curr_layer,
                                                  int start_chan,
                                                  float* result) {
    int in_row, in_col, in_chan, out_row, out_col, kern_row;
    unsigned j;

    const int a_height = curr_layer.inputs.rows;
    const int a_width = curr_layer.inputs.cols;
    const int a_pad = curr_layer.inputs.align_pad;
    const int a_padded_width = a_width + a_pad;

    const int result_height = curr_layer.outputs.rows;
    const int result_width = curr_layer.outputs.cols;
    const int result_pad = curr_layer.outputs.align_pad;
    const int result_padded_width = result_width + result_pad;

    const int k_width = curr_layer.weights.cols;
    const int k_pad = curr_layer.weights.align_pad;
    const int k_padded_width = k_width + k_pad;

    // Convolution control parameters.
    const int row_stride = curr_layer.stride.rows;
    const int in_col_stride = 1;
    const int k_col_stride = curr_layer.stride.cols;
    const int chan_stride = 1;
    const bool double_tp = k_width < DATAPATH_WIDTH;
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

    const int end_row = a_height - k_width + 1;
    const int end_col = (has_boundary_case ? input_fetches_per_row
                                           : input_fetches_per_row - 1);
    const int end_kern = k_width;
    const int end_chan = curr_layer.weights.height;

    VEC_ARRAY_3D(v8fp_t, _a, a, a_height, a_padded_width);
    VEC_ARRAY_3D(v8fp_t, _kernels, kernels, k_width, k_padded_width);
    VEC_ARRAY_3D(v8fp_t, _result, result, result_height, result_padded_width);

    int end_col_marker = (input_fetches_per_row - 1);
    int line_start = 0;  // Which vector element to commit to, per 8 packed activations.

    conv2d_chan:
    for (in_chan = 0; in_chan < end_chan; in_chan += chan_stride) {
        PRINT_MSG_V("Input channel %d\n", in_chan);
        out_row = 0;
        conv2d_row:
        for (in_row = 0; in_row < end_row; in_row += row_stride) {
            out_col = 0;
            line_start = 0;
            conv2d_col:
            for (in_col = 0; in_col < end_col; in_col += in_col_stride) {
                // Compute schedule.
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
                v8fp_t psums_0[1] = { (v8fp_t){ 0, 0, 0, 0, 0, 0, 0, 0 } };
                v8fp_t psums_1[1] = { (v8fp_t){ 0, 0, 0, 0, 0, 0, 0, 0 } };
                conv2d_kern_row:
                for (kern_row = 0; kern_row < end_kern; kern_row ++) {
                    v8fp_t weights_buffer = (v8fp_t){ 0 };
                    v8fp_t pipe0_shift_reg[2] = {
                        (v8fp_t){ 0, 0, 0, 0, 0, 0, 0, 0 },
                        (v8fp_t){ 0, 0, 0, 0, 0, 0, 0, 0 }
                    };
                    v8fp_t pipe1_shift_reg[2] = {
                        (v8fp_t){ 0, 0, 0, 0, 0, 0, 0, 0 },
                        (v8fp_t){ 0, 0, 0, 0, 0, 0, 0, 0 }
                    };

                    // Load activations into shift registers.
                    v8fp_t act_temp =
                            _a[in_chan + start_chan][in_row + kern_row][in_col];
                    pipe0_shift_reg[0] = act_temp;
                    pipe1_shift_reg[0] = act_temp;
                    if (!(has_boundary_case && in_col == end_col_marker)) {
                        act_temp = _a[in_chan + start_chan][in_row + kern_row]
                                     [in_col + 1];
                        pipe0_shift_reg[1] = act_temp;
                        pipe1_shift_reg[1] = act_temp;
                    }

                    PRINT_MSG_V("Shift registers after loading activations\n");
                    PRINT_DEBUG_V(&pipe0_shift_reg[0][0], 1, SHIFT_REG_SIZE, SHIFT_REG_SIZE);
                    PRINT_DEBUG_V(&pipe1_shift_reg[0][0], 1, SHIFT_REG_SIZE, SHIFT_REG_SIZE);

                    // Load weights into weights buffer, accounting for double tp
                    // mode.
                    v8fp_t wgt_temp = _kernels[in_chan][kern_row][0];
                    if (double_tp) {
                      conv2d_load_wgts_double_tp:
                      for (int w = 0; w < k_width; w++) {
                          weights_buffer[w] = wgt_temp[w];
                          weights_buffer[DATAPATH_WIDTH + w] = wgt_temp[w];
                      }
                    } else {
                      int bound = min2(k_width, VECTOR_SIZE);
                      conv2d_load_wgts_single_tp:
                      for (int w = 0; w < bound; w++) {
                          weights_buffer[w] = wgt_temp[w];
                      }
                    }

                    PRINT_MSG_V("Weights buffer\n");
                    PRINT_DEBUG_V(&weights_buffer[0], 1, VECTOR_SIZE, VECTOR_SIZE);

                    shift_reg_simd_lshift(pipe1_shift_reg, init_shamt);
                    PRINT_MSG_V("After initial shift of pipe1\n");
                    PRINT_DEBUG_V(&pipe1_shift_reg[0][0], 1, SHIFT_REG_SIZE, SHIFT_REG_SIZE);

                    // Primary datapath.
                    conv_macc_datapath_simd_fxp(weights_buffer,
                                                pipe0_shift_reg,
                                                pipe1_shift_reg,
                                                dp_shamt,
                                                dp0_iters,
                                                dp1_iters,
                                                psums_0,
                                                psums_1);
                }

                v8fp_t final_psums =
                        merge_psums_simd_fxp(psums_0[0], psums_1[0], double_tp);
                final_psums = activation_fun_simd_fxp(
                        final_psums, curr_layer.activation);

                conv2d_commit:
                for (j = line_start; j < line_start + total_outpx; j++)
                    _result[in_chan][out_row][out_col / VECTOR_SIZE][j] =
                            final_psums[j - line_start];
                out_col += total_outpx;
                line_start += total_outpx;
                if (out_col >= result_width)
                    out_col = 0;
                if (line_start >= VECTOR_SIZE)
                    line_start = 0;
            }
            PRINT_MSG_V("\nResult of row %d\n", out_row);
            PRINT_DEBUG_V(&_result[in_chan][out_row][0][0], 1, result_width, result_width);
            out_row++;
        }
    }
}
