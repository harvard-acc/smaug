#include <stdio.h>

#include "core/nnet_fwd_defs.h"
#include "core/smiv/params.h"
#include "core/smv/params.h"

// Perform a 3D convolution with one kernel on an image, with reduction in NHWC
// format. This is the vectorized implementation.
//
// Args:
//   a: 3D array, indexed as [row][col][channel].
//   kernels: A 3D kernel, indexed as [row][col][channel].
//   curr_layer: Layer (or partial layer) configuration.
//   kern_start: If the kernel array contains weights for multiple output
//      feature maps, start from this one.
//   result: a 3D array indexed as [channel][row][col].
//
// Returns:
//   The reduced 3D convolution values in result in NCHW format.
void convolution3d_smv_nhwc_vec_fxp(float* a,
                                    float* kernels,
                                    layer_t curr_layer,
                                    int kern_start,
                                    float* result) {
    int result_rows = curr_layer.outputs.rows;
    int result_cols = curr_layer.outputs.cols;
    int result_pad = curr_layer.outputs.align_pad;

    int k_cols = curr_layer.weights.cols;
    int k_rows = curr_layer.weights.rows;
    int k_height = curr_layer.weights.height;
    int k_pad = curr_layer.weights.align_pad;
    int k_stride = curr_layer.field_stride;

    int a_rows = curr_layer.inputs.rows;
    int a_cols = curr_layer.inputs.cols;
    int a_height = curr_layer.inputs.height;
    int a_pad = curr_layer.inputs.align_pad;

    int end_row = a_rows - k_rows + 1;
    int end_col = a_cols - k_cols + 1;

    int in_row, in_col;
    const int pe_depth = VECTOR_SIZE * NUM_MACC_INSTS;
    // If we have less than four channels, don't run the extra ones.
    const int kEffNumPeInsts = min2(curr_layer.outputs.height, NUM_PE_INSTS);

    VEC_ARRAY_3D(v8fp_t, _result, result, result_rows, result_cols + result_pad);
    VEC_ARRAY_4D(v8fp_t, _kernels, kernels, k_rows, k_cols, k_height + k_pad);
    VEC_ARRAY_3D(v8fp_t, _a, a, a_cols, a_height + a_pad);
    int num_chan_blocks = (k_height - 1) / pe_depth;

    k_col:
    for (int kern_row = 0; kern_row < k_rows; kern_row++) {  // Kernel rows
        k_row:
        for (int kern_col = 0; kern_col < k_cols; kern_col++) {  // Kernel cols
            // This loops over all the input channels in groups of VECTOR_SIZE
            // * NUM_MACC_INSTS.
            pe_iteration:
            for (int ifmap_iters = 0; ifmap_iters < num_chan_blocks + 1;
                 ifmap_iters++) {
                bool start_from_zero =
                        (kern_row == 0 && kern_col == 0 && ifmap_iters == 0);
                int ifmap_offset = (ifmap_iters * pe_depth) / VECTOR_SIZE;
                int out_i = 0;  // The result row.

                int max_ch_grp = NUM_MACC_INSTS;
                // this is just computing the remaining groups of
                // channels on the last iteration.
                if (ifmap_iters == num_chan_blocks) {
                    max_ch_grp = ((k_height -
                                   ifmap_iters * NUM_MACC_INSTS * VECTOR_SIZE) /
                                  VECTOR_SIZE) + 1;
                }

                // Load in all the weights at once before beginning the input
                // loop.
                v8fp_t kernel_reg[NUM_PE_INSTS][NUM_MACC_INSTS];
                const v8fp_t zero = (v8fp_t){ 0, 0, 0, 0, 0, 0, 0, 0 };
                load_kern_pe:
                for (int pe_id = 0; pe_id < kEffNumPeInsts; pe_id++) {
                    load_kern_mu:
                    for (int macc_idx = 0; macc_idx < NUM_MACC_INSTS;
                         macc_idx++) {
                        kernel_reg[pe_id][macc_idx] =
                                (macc_idx >= max_ch_grp)
                                        ? zero
                                        : _kernels[kern_start + pe_id][kern_row]
                                                  [kern_col]
                                                  [ifmap_offset + macc_idx];
                    }
                }

                conv3d_row:
                for (int out_row = 0; out_row < end_row; out_row += k_stride) {
                    int out_j = 0;  // The result col.

                    // We buffer all the partial sums into a vector register
                    // and write back to the scratchpad only when we've
                    // finished 8 partial sums.
                    v8fp_t results_buffer[NUM_PE_INSTS];

                    conv3d_col:
                    for (int out_col = 0; out_col < end_col;
                         out_col += k_stride) {
                        // Local Regs. These should always be sized the same (so
                        // NUM_PE_INSTS, rather than kNumEffPeInsts).
                        v8fp_t product_reg[NUM_PE_INSTS][NUM_MACC_INSTS];
                        v8fp_t act_reg[NUM_MACC_INSTS];
                        int res_buf_j = out_j % VECTOR_SIZE;
                        if (res_buf_j == 0) {
                            reload_results_buffer:
                            for (int i = 0; i < kEffNumPeInsts; i++) {
                                results_buffer[i] =
                                        start_from_zero
                                                ? zero
                                                : _result[kern_start + i][out_i]
                                                         [out_j / VECTOR_SIZE];
                            }
                        }
                        in_row = out_row + kern_row;
                        in_col = out_col + kern_col;

                        // Load in the activations first, then broadcast them
                        // to all the PEs.
                        load_act_mu:
                        for (int macc_idx = 0; macc_idx < NUM_MACC_INSTS;
                             macc_idx++) {
                            act_reg[macc_idx] =
                                    (macc_idx >= max_ch_grp)
                                            ? zero
                                            : _a[in_row][in_col]
                                                [ifmap_offset + macc_idx];
                        }

                        pe_groups:
                        for (int pe_id = 0; pe_id < kEffNumPeInsts; pe_id++) {
                            float accum_reg = 0;

                            mu_groups:
                            for (int macc_idx = 0; macc_idx < NUM_MACC_INSTS;
                                 macc_idx++) {
                                product_reg[pe_id][macc_idx] =
                                        kernel_reg[pe_id][macc_idx] *
                                        act_reg[macc_idx];
                            }
                            accum_reg = results_buffer[pe_id][res_buf_j];
                            reduction_adders:
                            for (int macc_idx = 0; macc_idx < NUM_MACC_INSTS;
                                 macc_idx++) {
                                reduction_inner:
                                for (int vec_i = 0; vec_i < VECTOR_SIZE;
                                     vec_i++) {
                                    accum_reg +=
                                            product_reg[pe_id][macc_idx][vec_i];
                                }
                            }
                            results_buffer[pe_id][res_buf_j] =
                                    accum_reg;
                        }

                        if ((out_j + 1) % VECTOR_SIZE == 0 ||
                            (out_j + 1) == result_cols) {
                            pe_commit:
                            for (int pe_id = 0; pe_id < kEffNumPeInsts;
                                 pe_id++) {
                                _result[kern_start + pe_id][out_i]
                                       [out_j / VECTOR_SIZE] =
                                               results_buffer[pe_id];
                            }
                        }
                        out_j++;
                    }
                    out_i++;
                    out_j = 0;
                }
            }
        }
    }
}
