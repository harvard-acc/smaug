// Functions to flatten a set of 3D input activations into a stack of row
// vectors.
//
// These will handle removing any additional zero-padding at the end of each
// image row (for data alignment) during the flattening and re-zeropadding the
// final row vector as appropriate.

#include <string.h>

#include "core/ref/flatten.h"
#include "utility/utility.h"

result_buf flatten_input_rowmajor(float* input,
                                  layer_t* layers,
                                  int lnum,
                                  float* result) {
    // Check if we actually need to do anything.
    if (layers[lnum - 1].outputs.align_pad == 0 &&
        layers[lnum].inputs.align_pad == 0)
        return input;

    // The "input" to im2row is the output of the previous layer.
    int input_rows = layers[lnum - 1].outputs.rows;
    int input_cols = layers[lnum - 1].outputs.cols;
    int input_height = layers[lnum - 1].outputs.height;
    int input_pad = layers[lnum - 1].outputs.align_pad;
    // The "output" padding of im2row is the padding required for the input to
    // the current layer.
    int output_pad = layers[lnum].inputs.align_pad;

    im2row(input, input_rows, input_cols, input_height, input_pad, output_pad,
           result);

    return result;
}

void im2row(float* input,
            int input_rows,
            int input_cols,
            int input_height,
            int input_pad,
            int output_pad,
            float* result) {
    int img, hgt, row;

    ARRAY_4D(float, _input, input, input_height, input_rows,
             input_cols + input_pad);

    for (img = 0; img < NUM_TEST_CASES; img++) {
        for (hgt = 0; hgt < input_height; hgt++) {
            for (row = 0; row < input_rows; row++) {
                memcpy(result, &_input[img][hgt][row][0], input_cols * sizeof(float));
                result += input_cols;
            }
        }
        memset((void*)result, 0, output_pad * sizeof(float));
        result += output_pad;
    }
}
