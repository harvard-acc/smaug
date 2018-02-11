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

    int k_cols = curr_layer.weights.cols;
    int k_rows = curr_layer.weights.rows;
    int k_height = curr_layer.weights.height;
    int k_pad = curr_layer.weights.align_pad;
    int k_stride = curr_layer.field_stride;

    int a_width = curr_layer.inputs.cols;
    int a_height = curr_layer.inputs.height;
    int a_pad = curr_layer.inputs.align_pad;

    int res_row, res_col;
    const int pe_depth = VECTOR_SIZE * NUM_MACC_INSTS;
    // Local Regs
    float product_reg[NUM_PE_INSTS][NUM_MACC_INSTS][VECTOR_SIZE];
    float act_reg[NUM_MACC_INSTS * VECTOR_SIZE];

    ARRAY_3D(float, _result, result, result_rows, result_cols + result_pad);
    ARRAY_4D(float, _kernels, kernels, k_cols, k_cols, k_height + k_pad);
    ARRAY_3D(float, _a, a, a_width, a_height + a_pad);
    int num_chan_blocks = (k_height - 1) / pe_depth;
    k_col:
    for (int kern_row = 0; kern_row < k_rows; kern_row++) {  // Kernel rows
        k_row:
        for (int kern_col = 0; kern_col < k_cols; kern_col++) {  // Kernel cols

            // This loops over all the input channels in groups of VECTOR_SIZE
            // * NUM_MACC_INSTS.
            pe_iteration:
            for (int pe_iters = 0; pe_iters < num_chan_blocks + 1; pe_iters++) {
                int pe_offset = pe_iters * pe_depth;
                conv3d_row:
                for (int out_row = 0; out_row < result_rows;
                     out_row += k_stride) {
                    conv3d_col:
                    for (int out_col = 0; out_col < result_cols;
                         out_col += k_stride) {
                        res_row = out_row + kern_row;
                        res_col = out_col + kern_col;

                        int max_ch_grp = NUM_MACC_INSTS;
                        // This is just computing the remaining groups of
                        // channels on the last iteration.
                        if (pe_iters == num_chan_blocks) {
                            max_ch_grp = ((k_height - pe_iters * NUM_MACC_INSTS *
                                                              VECTOR_SIZE) /
                                          VECTOR_SIZE) +
                                         1;
                        }
                        reset_act_mu:
                        for (int macc_idx = 0; macc_idx < NUM_MACC_INSTS;
                             macc_idx++) {
                            int macc_offset = macc_idx * VECTOR_SIZE;
                            reset_act_vector:
                            for (int vec_i = 0; vec_i < VECTOR_SIZE; vec_i++) {
                                int chan_idx = macc_offset + vec_i;
                                act_reg[chan_idx] = 0;
                            }
                        }
                        load_act_mu:
                        for (int macc_idx = 0; macc_idx < max_ch_grp;
                             macc_idx++) {
                            int macc_offset = macc_idx * VECTOR_SIZE;
                            load_act_vector:
                            for (int vec_i = 0; vec_i < VECTOR_SIZE; vec_i++) {
                                int chan_idx = macc_offset + vec_i;
                                act_reg[chan_idx] = _a[res_row][res_col]
                                                     [pe_offset + chan_idx];
                            }
                        }
                        pe_groups:
                        for (int pe_id = 0; pe_id < NUM_PE_INSTS; pe_id++) {
                            float accum_reg;
                            float kernel_reg[NUM_MACC_INSTS * VECTOR_SIZE];
                            reset_wt_mu:
                            for (int macc_idx = 0; macc_idx < NUM_MACC_INSTS;
                                 macc_idx++) {
                                int macc_offset = macc_idx * VECTOR_SIZE;
                                reset_wt_vector:
                                for (int vec_i = 0; vec_i < VECTOR_SIZE;
                                     vec_i++) {
                                    int chan_idx = macc_offset + vec_i;
                                    kernel_reg[chan_idx] = 0;
                                }
                            }
                            load_wt_mu:
                            for (int macc_idx = 0; macc_idx < max_ch_grp;
                                 macc_idx++) {
                                int macc_offset = macc_idx * VECTOR_SIZE;
                                load_wt_vector:
                                for (int vec_i = 0; vec_i < VECTOR_SIZE;
                                     vec_i++) {
                                    int chan_idx = macc_offset + vec_i;
                                    kernel_reg[chan_idx] =
                                            _kernels[kern_start + pe_id]
                                                    [kern_row][kern_col]
                                                    [pe_offset + chan_idx];
                                }
                            }
                            mu_groups:
                            for (int macc_idx = 0; macc_idx < NUM_MACC_INSTS;
                                 macc_idx++) {
                                int macc_offset = macc_idx * VECTOR_SIZE;
                                ch_groups:
                                for (int vec_i = 0; vec_i < VECTOR_SIZE;
                                     vec_i++) {
                                    int chan_idx = macc_offset + vec_i;
                                    product_reg[pe_id][macc_idx][vec_i] =
                                            kernel_reg[chan_idx] *
                                            act_reg[chan_idx];
                                }
                            }
                            if (kern_col == 0 && kern_row == 0 &&
                                pe_iters == 0) {
                                accum_reg = 0;
                            } else {
                                accum_reg = _result[kern_start + pe_id][out_row]
                                                   [out_col];
                            }
                            reduction_adders:
                            for (int macc_idx = 0; macc_idx < NUM_MACC_INSTS;
                                 macc_idx++) {
                                reduction_inner:
                                for (int vec_i = 0; vec_i < VECTOR_SIZE;
                                     vec_i++) {
                                    accum_reg =
                                            accum_reg +
                                            product_reg[pe_id][macc_idx][vec_i];
                                }
                            }
                            _result[kern_start + pe_id][out_row][out_col] =
                                    accum_reg;
                        }
                    }
                }
            }
        }
    }
}
