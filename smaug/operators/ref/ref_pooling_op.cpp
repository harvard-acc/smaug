#include <utility>

#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/pooling_op.h"

#ifdef __cplusplus
extern "C" {
#endif

/** \ingroup AladdinKernels
 * A Reference implementation of MaxPoolingOp on NCHW data, using a tree-based
 * maximum function.
 */
void ref_max_pooling_nchw_treemax(float* input,
                                  float* result,
                                  int img_num,
                                  int img_chans,
                                  int img_rows,
                                  int img_cols,
                                  int img_pad,
                                  int res_rows,
                                  int res_cols,
                                  int res_pad,
                                  int pool_row_size,
                                  int pool_col_size,
                                  int pool_row_stride,
                                  int pool_col_stride) {
    int total_pool_size = pool_row_size * pool_col_size;
    int end_row = img_rows - pool_row_size + 1;
    int end_col = img_cols - pool_col_size + 1;
    float elems[total_pool_size];
    int elem_idx;

    ARRAY_4D(float, _input, input, img_chans, img_rows, img_cols + img_pad);
    ARRAY_4D(float, _result, result, img_chans, res_rows, res_cols + res_pad);

    maxpool_input_num:
    for (int img = 0; img < img_num; img++) {
        maxpool_input_height:
        for (int h = 0; h < img_chans; h++) {
            int oi = 0;
            maxpool_input_rows:
            for (int i = 0; i < end_row; i += pool_row_stride) {
                int oj = 0;
                maxpool_input_cols:
                for (int j = 0; j < end_col; j += pool_col_stride) {
                    elem_idx = 0;
                    maxpool_tree_outer:
                    // Iterate over the pooling field.
                    for (int k = 0; k < pool_row_size; k++) {
                        maxpool_tree_inner:
                        for (int l = 0; l < pool_col_size; l++) {
                            elems[elem_idx] = _input[img][h][i+k][j+l];
                            elem_idx++;
                        }
                    }

                    float curr_max = 0;
                    if (total_pool_size == 4)
                        curr_max = max4(elems[0], elems[1], elems[2], elems[3]);
                    else if (total_pool_size == 9)
                        curr_max = max9(elems[0], elems[1], elems[2], elems[3],
                                        elems[4], elems[5], elems[6], elems[7],
                                        elems[8]);
                    else
                        assert(false && "Unsupported pooling size!");

                    _result[img][h][oi][oj] = curr_max;
                    oj++;
                }
                oi++;
                oj = 0;
            }
        }
    }
}

/** \ingroup AladdinKernels
 * A Reference implementation of MaxPoolingOp on NHWC data, using a tree-based
 * maximum function.
 */
void ref_max_pooling_nhwc_treemax(float* input,
                                  float* result,
                                  int img_num,
                                  int img_chans,
                                  int img_rows,
                                  int img_cols,
                                  int img_pad,
                                  int res_rows,
                                  int res_cols,
                                  int res_pad,
                                  int pool_row_size,
                                  int pool_col_size,
                                  int pool_row_stride,
                                  int pool_col_stride) {
    int total_pool_size = pool_row_size * pool_col_size;
    int end_row = img_rows - pool_row_size + 1;
    int end_col = img_cols - pool_col_size + 1;
    float elems[total_pool_size];
    int elem_idx;

    ARRAY_4D(float, _input, input, img_rows, img_cols, img_chans + img_pad);
    ARRAY_4D(float, _result, result, res_rows, res_cols, img_chans + res_pad);

    maxpool_input_num:
    for (int img = 0; img < img_num; img++) {
        maxpool_input_height:
        for (int h = 0; h < img_chans; h++) {
            int oi = 0;
            maxpool_input_rows:
            for (int i = 0; i < end_row; i += pool_row_stride) {
                int oj = 0;
                maxpool_input_cols:
                for (int j = 0; j < end_col; j += pool_col_stride) {
                    elem_idx = 0;
                    maxpool_tree_outer:
                    // Iterate over the pooling field.
                    for (int k = 0; k < pool_row_size; k++) {
                        maxpool_tree_inner:
                        for (int l = 0; l < pool_col_size; l++) {
                            elems[elem_idx] = _input[img][i+k][j+l][h];
                            elem_idx++;
                        }
                    }

                    float curr_max = 0;
                    if (total_pool_size == 4)
                        curr_max = max4(elems[0], elems[1], elems[2], elems[3]);
                    else if (total_pool_size == 9)
                        curr_max = max9(elems[0], elems[1], elems[2], elems[3],
                                        elems[4], elems[5], elems[6], elems[7],
                                        elems[8]);
                    else
                        assert(false && "Unsupported pooling size!");

                    _result[img][oi][oj][h] = curr_max;
                    oj++;
                }
                oi++;
                oj = 0;
            }
        }
    }
}

/** \ingroup AladdinKernels
 * A Reference implementation of MaxPoolingOp on NCHW data, using a loop-based
 * maximum function.
 */
void ref_max_pooling_nchw_itermax(float* input,
                                  float* result,
                                  int img_num,
                                  int img_chans,
                                  int img_rows,
                                  int img_cols,
                                  int img_pad,
                                  int res_rows,
                                  int res_cols,
                                  int res_pad,
                                  int pool_row_size,
                                  int pool_col_size,
                                  int pool_row_stride,
                                  int pool_col_stride) {
    int end_row = img_rows - pool_row_size + 1;
    int end_col = img_cols - pool_col_size + 1;
    ARRAY_4D(float, _input, input, img_chans, img_rows, img_cols + img_pad);
    ARRAY_4D(float, _result, result, img_chans, res_rows, res_cols + res_pad);

    maxpool_input_num:
    for (int img = 0; img < img_num; img++) {
        maxpool_input_height:
        for (int h = 0; h < img_chans; h++) {
            int oi = 0;
            maxpool_input_rows:
            for (int i = 0; i < end_row; i += pool_row_stride) {
                int oj = 0;
                maxpool_input_cols:
                for (int j = 0; j < end_col; j += pool_col_stride) {
                    float curr_max = -FLT_MAX;
                    maxpool_iter_outer:
                    for (int k = 0; k < pool_row_size; k++) {
                        maxpool_iter_inner:
                        for (int l = 0; l < pool_col_size; l++) {
                            float in_val = _input[img][h][i+k][j+l];
                            curr_max = max2(in_val, curr_max);
                        }
                    }

                    _result[img][h][oi][oj] = curr_max;
                    oj++;
                }
                oi++;
                oj = 0;
            }
        }
    }
}

/** \ingroup AladdinKernels
 * A Reference implementation of MaxPoolingOp on NHWC data, using a loop-based
 * maximum function.
 */
void ref_max_pooling_nhwc_itermax(float* input,
                                  float* result,
                                  int img_num,
                                  int img_chans,
                                  int img_rows,
                                  int img_cols,
                                  int img_pad,
                                  int res_rows,
                                  int res_cols,
                                  int res_pad,
                                  int pool_row_size,
                                  int pool_col_size,
                                  int pool_row_stride,
                                  int pool_col_stride) {
    int end_row = img_rows - pool_row_size + 1;
    int end_col = img_cols - pool_col_size + 1;
    ARRAY_4D(float, _input, input, img_rows, img_cols, img_chans + img_pad);
    ARRAY_4D(float, _result, result, res_rows, res_cols, img_chans + res_pad);

    maxpool_input_num:
    for (int img = 0; img < img_num; img++) {
        maxpool_input_height:
        for (int h = 0; h < img_chans; h++) {
            int oi = 0;
            maxpool_input_rows:
            for (int i = 0; i < end_row; i += pool_row_stride) {
                int oj = 0;
                maxpool_input_cols:
                for (int j = 0; j < end_col; j += pool_col_stride) {
                    float curr_max = -FLT_MAX;
                    maxpool_iter_outer:
                    for (int k = 0; k < pool_row_size; k++) {
                        maxpool_iter_inner:
                        for (int l = 0; l < pool_col_size; l++) {
                            float in_val = _input[img][i+k][j+l][h];
                            curr_max = max2(in_val, curr_max);
                        }
                    }

                    _result[img][oi][oj][h] = curr_max;
                    oj++;
                }
                oi++;
                oj = 0;
            }
        }
    }
}

/** \ingroup AladdinKernels
 * A Reference implementation of AvgPoolingOp on NCHW data.
 */
void ref_avg_pooling_nchw(float* input,
                          float* result,
                          int img_num,
                          int img_chans,
                          int img_rows,
                          int img_cols,
                          int img_pad,
                          int res_rows,
                          int res_cols,
                          int res_pad,
                          int pool_row_size,
                          int pool_col_size,
                          int pool_row_stride,
                          int pool_col_stride) {
    int end_row = img_rows - pool_row_size + 1;
    int end_col = img_cols - pool_col_size + 1;
    ARRAY_4D(float, _input, input, img_chans, img_rows, img_cols + img_pad);
    ARRAY_4D(float, _result, result, img_chans, res_rows, res_cols + res_pad);
    float recip_total_size = 1.0 / (pool_row_size * pool_col_size);

    maxpool_input_num:
    for (int img = 0; img < img_num; img++) {
        maxpool_input_height:
        for (int h = 0; h < img_chans; h++) {
            int oi = 0;
            maxpool_input_rows:
            for (int i = 0; i < end_row; i += pool_row_stride) {
                int oj = 0;
                maxpool_input_cols:
                for (int j = 0; j < end_col; j += pool_col_stride) {
                    float curr_sum = 0;
                    avgpool_iter_outer:
                    for (int k = 0; k < pool_row_size; k++) {
                        avgpool_iter_inner:
                        for (int l = 0; l < pool_col_size; l++) {
                            curr_sum += _input[img][h][i+k][j+l];
                        }
                    }

                    _result[img][h][oi][oj] = curr_sum * recip_total_size;
                    oj++;
                }
                oi++;
                oj = 0;
            }
        }
    }
}

/** \ingroup AladdinKernels
 * AladdinKernels Reference implementation of AvgPoolingOp on NHWC data.
 */
void ref_avg_pooling_nhwc(float* input,
                          float* result,
                          int img_num,
                          int img_chans,
                          int img_rows,
                          int img_cols,
                          int img_pad,
                          int res_rows,
                          int res_cols,
                          int res_pad,
                          int pool_row_size,
                          int pool_col_size,
                          int pool_row_stride,
                          int pool_col_stride) {
    int end_row = img_rows - pool_row_size + 1;
    int end_col = img_cols - pool_col_size + 1;
    ARRAY_4D(float, _input, input, img_rows, img_cols, img_chans + img_pad);
    ARRAY_4D(float, _result, result, res_rows, res_cols, img_chans + res_pad);
    float recip_total_size = 1.0 / (pool_row_size * pool_col_size);

    maxpool_input_num:
    for (int img = 0; img < img_num; img++) {
        maxpool_input_height:
        for (int h = 0; h < img_chans; h++) {
            int oi = 0;
            maxpool_input_rows:
            for (int i = 0; i < end_row; i += pool_row_stride) {
                int oj = 0;
                maxpool_input_cols:
                for (int j = 0; j < end_col; j += pool_col_stride) {
                    float curr_sum = 0;
                    avgpool_iter_outer:
                    for (int k = 0; k < pool_row_size; k++) {
                        avgpool_iter_inner:
                        for (int l = 0; l < pool_col_size; l++) {
                            curr_sum += _input[img][i+k][j+l][h];
                        }
                    }

                    _result[img][oi][oj][h] = curr_sum * recip_total_size;
                    oj++;
                }
                oi++;
                oj = 0;
            }
        }
    }
}

#ifdef __cplusplus
}
#endif

namespace smaug {

template <>
void MaxPoolingOp<ReferenceBackend>::run() {
    auto input = getInput(Inputs);
    auto output = getOutput(Outputs);
    const TensorShape& inputShape = input->getShape();
    const TensorShape& outputShape = output->getShape();

    bool isNCHW = input->getShape().getLayout() == NCHW;
    bool useTreeMax = (poolingRowSize <= 3 && poolingRowSize == poolingColSize);
    auto func = isNCHW ? (useTreeMax ? ref_max_pooling_nchw_treemax
                                     : ref_max_pooling_nchw_itermax)
                       : (useTreeMax ? ref_max_pooling_nhwc_treemax
                                     : ref_max_pooling_nhwc_itermax);
    int poolRowSize, poolColSize, poolRowStride, poolColStride;
    std::tie(poolRowSize, poolColSize) = getPoolingSize();
    std::tie(poolRowStride, poolColStride) = getPoolingStride();

    float* inputData = input->data<float>();
    float* outputData = output->data<float>();
    mapArrayToAccel(ref::kPoolingHw, "input", inputData,
                    inputShape.storageSize() * sizeof(float));
    mapArrayToAccel(ref::kPoolingHw, "result", outputData,
                    outputShape.storageSize() * sizeof(float));
    int rowIdx = isNCHW ? 2 : 1;
    int colIdx = isNCHW ? 3 : 2;
    int chanIdx = isNCHW ? 1 : 3;
    invokeKernel(ref::kPoolingHw, func, inputData, outputData, inputShape[0],
                 inputShape[chanIdx], inputShape[rowIdx], inputShape[colIdx],
                 inputShape.getPadding(3), outputShape[rowIdx],
                 outputShape[colIdx], outputShape.getPadding(3), poolRowSize,
                 poolColSize, poolRowStride, poolColStride);
}

template <>
void AvgPoolingOp<ReferenceBackend>::run() {
    auto input = getInput(Inputs);
    auto output = getOutput(Outputs);
    const TensorShape& inputShape = input->getShape();
    const TensorShape& outputShape = output->getShape();

    bool isNCHW = input->getShape().getLayout() == NCHW;
    auto func = isNCHW ? ref_avg_pooling_nchw : ref_avg_pooling_nhwc;
    int poolRowSize, poolColSize, poolRowStride, poolColStride;
    std::tie(poolRowSize, poolColSize) = getPoolingSize();
    std::tie(poolRowStride, poolColStride) = getPoolingStride();

    float* inputData = input->data<float>();
    float* outputData = output->data<float>();
    mapArrayToAccel(ref::kPoolingHw, "input", inputData,
                    inputShape.storageSize() * sizeof(float));
    mapArrayToAccel(ref::kPoolingHw, "result", outputData,
                    outputShape.storageSize() * sizeof(float));
    int rowIdx = isNCHW ? 2 : 1;
    int colIdx = isNCHW ? 3 : 2;
    int chanIdx = isNCHW ? 1 : 3;
    invokeKernel(ref::kPoolingHw, func, inputData, outputData, inputShape[0],
                 inputShape[chanIdx], inputShape[rowIdx], inputShape[colIdx],
                 inputShape.getPadding(3), outputShape[rowIdx],
                 outputShape[colIdx], outputShape.getPadding(3), poolRowSize,
                 poolColSize, poolRowStride, poolColStride);
}

}  // namespace smaug

