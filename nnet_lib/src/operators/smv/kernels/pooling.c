#include <stdio.h>
#include <float.h>

#include "operators/common.h"
#include "params.h"
#include "load_store_fp16_data.h"

#ifdef __cplusplus
extern "C" {
#endif

// A max-pooling operation on SMV with NHWC format. This is the
// vectorized implementation.
//
// Args:
//   host_inputs: Host inputs buffer in NHWC.
//   host_results: Host results buffer in NHWC.
//   inputs: Local inputs buffer in NHWC.
//   results: Local results buffer in NHWC.
//   inputs_dims: Dimensions of the inputs.
//   weights_dims: Dimensions of the weights.
//   results_dims: Dimensions of the results.
//   inputs_pad: Align padding size on the channel dimension of the
//       inputs.
//   results_pad: Align padding size on the channel dimension of the results.
//   pool_rows: Row size of the pooling function.
//   pool_cols: Column size of the pooling function.
//   row_stride: Stride size on the row dimension.
//   col_stride: Stride size on the col dimension.
//   ofmap_start: If the results contains more channels than the inputs, start
//       from this one. Otherwise this should always be zero.
void smv_maxpooling_nhwc_vec_fxp(float16* host_inputs,
                                 float16* host_results,
                                 float* inputs,
                                 float* results,
                                 int inputs_dims[4],
                                 int results_dims[4],
                                 int inputs_pad,
                                 int results_pad,
                                 int pool_rows,
                                 int pool_cols,
                                 int row_stride,
                                 int col_stride,
                                 int ofmap_start) {
    int a_rows = inputs_dims[1];
    int a_cols = inputs_dims[2];
    int a_height = inputs_dims[3];
    int a_pad = inputs_pad;
    int inputs_size = inputs_dims[0] * a_rows * a_cols * (a_height + a_pad);

    int results_rows = results_dims[1];
    int results_cols = results_dims[2];
    int results_height = results_dims[3];
    int results_size = results_dims[0] * results_rows * results_cols *
                       (results_height + results_pad);

    int chan_groups = FRAC_CEIL(a_height, VECTOR_SIZE);
    int ofmap_start_grp = ofmap_start / VECTOR_SIZE;
    int end_row = a_rows - pool_rows + 1;
    int end_col = a_cols - pool_cols + 1;

    // TODO: Support input batches.
    VEC_ARRAY_3D(v8fp_t, _a, inputs, a_cols, a_height + a_pad);
    VEC_ARRAY_3D(v8fp_t,
                 _results,
                 results,
                 results_cols,
                 results_height + results_pad);

    // Load inputs.
    host_load_fp16(inputs, host_inputs, inputs_size, 0, 0);

    int out_row = 0;
    maxpool_input_row:
    for (int row = 0; row < end_row; row += row_stride) {
        int out_col = 0;
        maxpool_input_col:
        for (int col = 0; col < end_col; col += col_stride) {
            maxpool_chan_grp:
            for (int chan_grp = 0; chan_grp < chan_groups; chan_grp++) {
                v8fp_t curr_results = {
                    -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX,
                    -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX
                };
                maxpool_pool_row:
                for (int pool_i = 0; pool_i < pool_rows; pool_i++) {
                    maxpool_pool_col:
                    for (int pool_j = 0; pool_j < pool_cols; pool_j++) {
                        v8fp_t next_pixels =
                                _a[row + pool_i][col + pool_j][chan_grp];
                        maxpool_compare:
                        for (int px = 0; px < VECTOR_SIZE; px++) {
                            if (curr_results[px] < next_pixels[px])
                                curr_results[px] = next_pixels[px];
                        }
                    }
                }
                // Commit.
                _results[out_row][out_col][ofmap_start_grp + chan_grp] =
                        curr_results;
            }
            out_col++;
        }
        out_row++;
    }

    // Store results to the host memory if needed.
    if (ofmap_start + a_height == results_height)
        host_store_fp16(results, host_results, results_size, 0, 0);
}

// An average-pooling operation on SMV with NHWC format. This is the
// vectorized implementation.
//
// Args:
//   host_inputs: Host inputs buffer in NHWC.
//   host_results: Host results buffer in NHWC.
//   inputs: Local inputs buffer in NHWC.
//   results: Local results buffer in NHWC.
//   inputs_dims: Dimensions of the inputs.
//   weights_dims: Dimensions of the weights.
//   results_dims: Dimensions of the results.
//   inputs_pad: Align padding size on the channel dimension of the
//       inputs.
//   results_pad: Align padding size on the channel dimension of the results.
//   pool_rows: Row size of the pooling function.
//   pool_cols: Column size of the pooling function.
//   row_stride: Stride size on the row dimension.
//   col_stride: Stride size on the col dimension.
//   ofmap_start: If the results contains more channels than the inputs, start
//       from this one. Otherwise this should always be zero.
// This requires a blocked channel data format (GNHWC), where G = channels/8,
// and the last dimension = chans = 8. The last dimension MUST be 8.
// This supports arbitrary pooling sizes and strides.
void smv_avgpooling_nhwc_vec_fxp(float16* host_inputs,
                                 float16* host_results,
                                 float* inputs,
                                 float* results,
                                 int inputs_dims[4],
                                 int results_dims[4],
                                 int inputs_pad,
                                 int results_pad,
                                 int pool_rows,
                                 int pool_cols,
                                 int row_stride,
                                 int col_stride,
                                 int ofmap_start) {
    int a_rows = inputs_dims[1];
    int a_cols = inputs_dims[2];
    int a_height = inputs_dims[3];
    int a_pad = inputs_pad;
    int inputs_size = inputs_dims[0] * a_rows * a_cols * (a_height + a_pad);

    int results_rows = results_dims[1];
    int results_cols = results_dims[2];
    int results_height = results_dims[3];
    int results_size = results_dims[0] * results_rows * results_cols *
                       (results_height + results_pad);

    int chan_groups = FRAC_CEIL(a_height, VECTOR_SIZE);
    int ofmap_start_grp = ofmap_start / VECTOR_SIZE;
    int end_row = a_rows - pool_rows + 1;
    int end_col = a_cols - pool_cols + 1;

    float scale = 1.0 / (pool_rows * pool_cols);
    v8fp_t scale_vec = {
        scale, scale, scale, scale, scale, scale, scale, scale
    };

    // TODO: Support input batches.
    VEC_ARRAY_3D(v8fp_t, _a, inputs, a_cols, a_height + a_pad);
    VEC_ARRAY_3D(v8fp_t,
                 _results,
                 results,
                 results_cols,
                 results_height + results_pad);

    // Load inputs.
    host_load_fp16(inputs, host_inputs, inputs_size, 0, 0);

    int out_row = 0;
    avgpool_input_row:
    for (int row = 0; row < end_row; row += row_stride) {
        int out_col = 0;
        avgpool_input_col:
        for (int col = 0; col < end_col; col += col_stride) {
            avgpool_chan_grp:
            for (int chan_grp = 0; chan_grp < chan_groups; chan_grp++) {
                v8fp_t curr_results = {0, 0, 0, 0, 0, 0, 0, 0};
                avgpool_pool_row:
                for (int pool_i = 0; pool_i < pool_rows; pool_i++) {
                    avgpool_pool_col:
                    for (int pool_j = 0; pool_j < pool_cols; pool_j++) {
                        curr_results +=
                                _a[row + pool_i][col + pool_j][chan_grp];
                    }
                }
                // Commit.
                _results[out_row][out_col][ofmap_start_grp + chan_grp] =
                        curr_results * scale_vec;
            }
            out_col++;
        }
        out_row++;
    }

    // Store results to the host memory if needed.
    if (ofmap_start + a_height == results_height)
        host_store_fp16(results, host_results, results_size, 0, 0);
}

#ifdef __cplusplus
}  // extern "C"
#endif
