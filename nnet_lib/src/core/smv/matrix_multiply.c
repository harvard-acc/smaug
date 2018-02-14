#include <assert.h>

#include "core/smiv/activation_functions_simd.h"
#include "core/smv/impls.h"
#include "utility/utility.h"
#include "nnet_fwd.h"

/*
* After transposition:
*
* weight cols (originally rows) --->
*
* rows  [[---][---][---]]
*  |    [[---][---][---]]
*       [[---][---][---]]
*  v    [[---][---][---]]
*
*  Each [---] represents an 8-wide vector. This inner product executes a 32-way
*  MACC -- 4 such 8-wide vectors -- per PE, and 4 PEs, where each PE is
*  assigned a row in in the transposed matrix. It continues across each row of
*  the weights until the complete output pixel is finished (output stationary).
*
*  No biases are added.
*/
void matrix_multiply_transpose_smv_nobatch_vec_fxp(float* a,
                                                   float* b,
                                                   int a_height,
                                                   int b_width,
                                                   int b_height,
                                                   int a_pad,
                                                   activation_type act_func,
                                                   int result_start,
                                                   float* result) {
    int a_width = b_width;
    ASSERT(b_width % VECTOR_SIZE == 0 &&
           "Width of weights must be a multiple of VECTOR_SIZE!");
    int b_width_vec = b_width / VECTOR_SIZE;

    v8fp_t zero = (v8fp_t){ 0, 0, 0, 0, 0, 0, 0, 0 };
    VEC_ARRAY_2D(v8fp_t, _a, a, a_width + a_pad);
    VEC_ARRAY_2D(v8fp_t, _b, b, b_width);
    VEC_ARRAY_2D(v8fp_t, _result, result, b_width);

    input_act:
    for (int input_act = 0; input_act < a_height; input_act++) {
        int psum_offset = 0;
        v8fp_t partial_sums = zero;
        wgt_row:
        for (int wgt_row = 0; wgt_row < b_height; wgt_row += NUM_PE_INSTS) {
            if (wgt_row % VECTOR_SIZE == 0)
                partial_sums = zero;

            wgt_col:
            for (int wgt_col = 0; wgt_col < b_width_vec; wgt_col+=NUM_MACC_INSTS) {
                v8fp_t activation_reg[NUM_MACC_INSTS];
                act_reg_load:
                for (int act_vec = 0; act_vec < NUM_MACC_INSTS; act_vec++) {
                    int act_col = wgt_col + act_vec;
                    activation_reg[act_vec] =
                            act_col >= b_width_vec ? zero : _a[input_act][act_col];
                }

                pe_insts:
                for (int pe_id = 0; pe_id < NUM_PE_INSTS; pe_id++) {
                    v8fp_t weights_reg[NUM_MACC_INSTS];
                    wgt_reg_load:
                    for (int macc_idx = 0; macc_idx < NUM_MACC_INSTS;
                         macc_idx++) {
                        int pe_row = wgt_row + pe_id;
                        int this_wgt_col = wgt_col + macc_idx;
                        weights_reg[macc_idx] =
                                (pe_row >= b_height ||
                                 this_wgt_col >= b_width_vec)
                                        ? zero
                                        : _b[pe_row][wgt_col + macc_idx];
                    }

                    v8fp_t product_reg[NUM_MACC_INSTS];
                    core_mul:
                    for (int macc_idx = 0; macc_idx < NUM_MACC_INSTS; macc_idx++) {
                        product_reg[macc_idx] = activation_reg[macc_idx] *
                                                weights_reg[macc_idx];
                    }

                    float accum_reg = 0;
                    reduce_mu:
                    for (int macc_idx = 0; macc_idx < NUM_MACC_INSTS; macc_idx++) {
                        for (int vec_i = 0; vec_i < VECTOR_SIZE; vec_i++) {
                            accum_reg += product_reg[macc_idx][vec_i];
                        }
                    }
                    partial_sums[psum_offset + pe_id] += accum_reg;
                }
            }
            int next_wgt_row = wgt_row + NUM_PE_INSTS;
            if (next_wgt_row % VECTOR_SIZE ==  0 || next_wgt_row >= b_height) {
                _result[input_act][(result_start + wgt_row) / VECTOR_SIZE] =
                        partial_sums;
                psum_offset = 0;
            } else {
                psum_offset += NUM_PE_INSTS;
            }
        }
    }
}
