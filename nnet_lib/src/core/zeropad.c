#include "nnet_fwd.h"

#include "zeropad.h"

// Zeropad each image in @a by layer.c_padding zeros.
//
// a is a 4D matrix of flattened input feature maps.
//
// The result is placed in @result, which is an array that is assumed to be
// large enough for this operation.
void copy_zeropad(float* a, layer_t curr_layer, float* result) {
    int ni;

    // "input_rows" and "input_cols" are the dimensions of the data AFTER
    // zeropadding because this is considered as the "input" to the convolution
    // itself.
    int pad = curr_layer.c_padding;
    int a_rows = curr_layer.inputs.rows - 2 * pad;
    int a_cols = curr_layer.inputs.cols - 2 * pad;
    int a_height = curr_layer.inputs.height;
    int a_data_pad = curr_layer.outputs.align_pad;
    int r_rows = curr_layer.inputs.rows;
    int r_cols = curr_layer.inputs.cols;
    int r_data_pad = curr_layer.inputs.align_pad;

copy_zeropad_per_image:
    for (ni = 0; ni < NUM_TEST_CASES; ni++) {
        copy_zeropad_image3d(a, pad, ni, a_rows, a_cols, a_height, a_data_pad,
                             result, r_rows, r_cols, r_data_pad);
    }
}

void copy_zeropad_image3d(float* a,
                          int pad,
                          int img,
                          int a_rows,
                          int a_cols,
                          int a_hgt,
                          int a_data_pad,
                          float* result,
                          int r_rows,
                          int r_cols,
                          int r_data_pad) {
    int h, i, j;

    ARRAY_4D(float, _a, a, a_hgt, a_rows, a_cols + a_data_pad);
    ARRAY_4D(float, _result, result, a_hgt, r_rows, r_cols + r_data_pad);

    copy_zeropad_height:
    for (h = 0; h < a_hgt; h++) {
        copy_zeropad_first_rows:
        for (i = 0; i < pad; i++) {
            copy_zeropad_first_cols:
            for (j = 0; j < r_cols + r_data_pad; j++) {
                _result[img][h][i][j] = 0;
            }
        }

        copy_zeropad_left:
        for (i = pad; i < a_rows + pad; i++) {
            copy_zeropad_left_cols:
            for (j = 0; j < pad; j++) {
                _result[img][h][i][j] = 0;
            }
            // Copy the original array.
            copy_zeropad_copy_cols:
            for (j = pad; j < a_cols + pad; j++) {
                _result[img][h][i][j] = _a[img][h][i-pad][j-pad];
            }
            copy_zeropad_right_cols:
            for (j = a_cols + pad; j < r_cols; j++) {
                _result[img][h][i][j] = 0;
            }
            copy_zeropad_data_pad:
            for (j = r_cols; j < r_cols + r_data_pad; j++) {
                _result[img][h][i][j] = 0;
            }
        }

        copy_zeropad_last:
        for (i = a_rows + pad; i < r_rows; i++) {
            copy_zeropad_last_cols:
            for (j = 0; j < r_cols + r_data_pad; j++) {
                _result[img][h][i][j] = 0;
            }
        }
    }
}
