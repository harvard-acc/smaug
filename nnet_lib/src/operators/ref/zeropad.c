#include "nnet_fwd.h"

#include "core/ref/zeropad.h"

// Zeropad each image in @a by layer.c_padding zeros.
//
// a is a 4D matrix of flattened input feature maps.
//
// The result is placed in @result, which is an array that is assumed to be
// large enough for this operation.
void copy_zeropad(float* a, layer_t* layers, int lnum, float* result) {
    int ni;

    layer_t curr_layer = layers[lnum];
    layer_t prev_layer = layers[lnum - 1];
    // "input_rows" and "input_cols" are the dimensions of the data AFTER
    // zeropadding because this is considered as the "input" to the convolution
    // itself.
    padding pad = curr_layer.pad;
    int a_rows = prev_layer.outputs.rows;
    int a_cols = prev_layer.outputs.cols;
    int a_height = prev_layer.outputs.height;
    int a_data_pad = prev_layer.outputs.align_pad;
    int r_rows = curr_layer.inputs.rows;
    int r_cols = curr_layer.inputs.cols;
    int r_data_pad = curr_layer.inputs.align_pad;

    copy_zeropad_per_image:
    for (ni = 0; ni < NUM_TEST_CASES; ni++) {
        copy_zeropad_image3d(a, &pad, ni, a_rows, a_cols, a_height, a_data_pad,
                             result, r_rows, r_cols, r_data_pad);
    }
}

void copy_zeropad_image3d(float* a,
                          padding* pad,
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
        for (i = 0; i < pad->top; i++) {
            copy_zeropad_first_cols:
            for (j = 0; j < r_cols + r_data_pad; j++) {
                _result[img][h][i][j] = 0;
            }
        }

        copy_zeropad_left:
        for (i = pad->top; i < a_rows + pad->top; i++) {
            copy_zeropad_left_cols:
            for (j = 0; j < pad->left; j++) {
                _result[img][h][i][j] = 0;
            }
            // Copy the original array.
            copy_zeropad_copy_cols:
            for (j = pad->left; j < a_cols + pad->left; j++) {
                _result[img][h][i][j] =
                        _a[img][h][i - pad->left][j - pad->left];
            }
            copy_zeropad_right_cols:
            for (j = a_cols + pad->left; j < r_cols; j++) {
                _result[img][h][i][j] = 0;
            }
            copy_zeropad_data_pad:
            for (j = r_cols; j < r_cols + r_data_pad; j++) {
                _result[img][h][i][j] = 0;
            }
        }

        copy_zeropad_last:
        for (i = a_rows + pad->top; i < r_rows; i++) {
            copy_zeropad_last_cols:
            for (j = 0; j < r_cols + r_data_pad; j++) {
                _result[img][h][i][j] = 0;
            }
        }
    }
}
