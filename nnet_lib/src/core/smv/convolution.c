#include <stdio.h>

#include "core/nnet_fwd_defs.h"
#include "core/smv/params.h"

// Perform a 3D convolution with one kernel on an image, with reduction in NHWC
// format
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
void convolution3d_smv_nhwc_fxp(float* a,
                                float* kernels,
                                layer_t curr_layer,
                                int kern_start,
                                float* result) {
    int result_rows = curr_layer.outputs.rows;
    int result_cols = curr_layer.outputs.cols;
    int result_pad = curr_layer.outputs.align_pad;

    int k_rows = curr_layer.weights.rows;
    int k_cols = curr_layer.weights.cols;
    int k_pad = curr_layer.weights.align_pad;
    int k_height = curr_layer.weights.height;
    int row_stride = curr_layer.stride.rows;
    int col_stride = curr_layer.stride.cols;

    int a_rows = curr_layer.inputs.rows;
    int a_cols = curr_layer.inputs.cols;
    int a_height = curr_layer.inputs.height;
    int a_pad = curr_layer.inputs.align_pad;

    padding pad = curr_layer.pad;
    int end_row = a_rows + pad.top + pad.bottom - k_rows + 1;
    int end_col = a_cols + pad.left + pad.right - k_cols + 1;

    int valid_row_end = a_rows - 1;
    int valid_col_end = a_cols - 1;

    int in_row, in_col;
    const int pe_depth = VECTOR_SIZE * NUM_MACC_INSTS;
    // If we have less than four channels, don't run the extra ones.
    const int kEffNumPeInsts = min2(curr_layer.outputs.height - kern_start, NUM_PE_INSTS);

    ARRAY_3D(float, _result, result, result_rows, result_cols + result_pad);
    ARRAY_4D(float, _kernels, kernels, k_rows, k_cols, k_height + k_pad);
    ARRAY_3D(float, _a, a, a_cols, a_height + a_pad);
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
                int ifmap_offset = ifmap_iters * pe_depth;
                int out_i = 0;  // The result row.

                int max_ch_grp = NUM_MACC_INSTS;
                // this is just computing the remaining groups of
                // channels on the last iteration.
                if (ifmap_iters == num_chan_blocks) {
                    max_ch_grp = FRAC_CEIL(
                            (k_height - ifmap_iters * pe_depth), VECTOR_SIZE);
                }

                // Load in all the weights at once before beginning the input
                // loop.
                float kernel_reg[NUM_PE_INSTS][NUM_MACC_INSTS][VECTOR_SIZE];
                load_kern_pe:
                for (int pe_id = 0; pe_id < kEffNumPeInsts; pe_id++) {
                    load_kern_mu:
                    for (int macc_idx = 0; macc_idx < NUM_MACC_INSTS;
                         macc_idx++) {
                        int macc_offset = macc_idx * VECTOR_SIZE;
                        load_kern_vector:
                        for (int vec_i = 0; vec_i < VECTOR_SIZE; vec_i++) {
                            int chan_idx = macc_offset + vec_i;
                            float value =
                                    (macc_idx >= max_ch_grp)
                                            ? 0
                                            : _kernels[kern_start + pe_id]
                                                      [kern_row][kern_col]
                                                      [ifmap_offset + chan_idx];
                            kernel_reg[pe_id][macc_idx][vec_i] = value;
                        }
                    }
                }

                conv3d_row:
                for (int out_row = 0; out_row < end_row; out_row += row_stride) {
                    int out_j = 0;  // The result col.

                    conv3d_col:
                    for (int out_col = 0; out_col < end_col;
                         out_col += col_stride) {
                        // Local Regs. These should always be sized the same (so
                        // NUM_PE_INSTS, rather than kNumEffPeInsts).
                        float product_reg[NUM_PE_INSTS][NUM_MACC_INSTS]
                                         [VECTOR_SIZE];
                        float act_reg[NUM_MACC_INSTS][VECTOR_SIZE];
                        in_row = out_row - pad.top + kern_row;
                        in_col = out_col - pad.left + kern_col;
                        bool in_padding_row =
                                in_row < 0 || in_row > valid_row_end;
                        bool in_padding_col =
                                in_col < 0 || in_col > valid_col_end;

                        // Load in the activations first, then broadcast them
                        // to all the PEs.
                        load_act_mu:
                        for (int macc_idx = 0; macc_idx < NUM_MACC_INSTS;
                             macc_idx++) {
                            int macc_offset = macc_idx * VECTOR_SIZE;
                            bool is_padding = in_padding_row ||
                                              in_padding_col ||
                                              macc_idx >= max_ch_grp;
                            load_act_vector:
                            for (int vec_i = 0; vec_i < VECTOR_SIZE; vec_i++) {
                                int chan_idx = macc_offset + vec_i;
                                float value =
                                        (is_padding)
                                                ? 0
                                                : _a[in_row][in_col]
                                                    [ifmap_offset + chan_idx];
                                act_reg[macc_idx][vec_i] = value;
                            }
                        }

                        pe_groups:
                        for (int pe_id = 0; pe_id < kEffNumPeInsts; pe_id++) {
                            float accum_reg = 0;

                            mu_groups:
                            for (int macc_idx = 0; macc_idx < NUM_MACC_INSTS;
                                 macc_idx++) {
                                ch_groups:
                                for (int vec_i = 0; vec_i < VECTOR_SIZE;
                                     vec_i++) {
                                    product_reg[pe_id][macc_idx][vec_i] =
                                            kernel_reg[pe_id][macc_idx][vec_i] *
                                            act_reg[macc_idx][vec_i];
                                }
                            }
                            if (kern_col == 0 && kern_row == 0 &&
                                ifmap_iters == 0) {
                                accum_reg = 0;
                            } else {
                                accum_reg = _result[kern_start + pe_id]
                                                         [out_i][out_j];
                            }
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
                            _result[kern_start + pe_id][out_i][out_j] =
                                    accum_reg;
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
