// Functions to flatten a set of 3D input activations into a stack of row
// vectors.
//
// These will handle removing any additional zero-padding at the end of each
// image row (for data alignment) during the flattening and re-zeropadding the
// final row vector as appropriate.

#include <string.h>

#include "utility/utility.h"
#include "nnet_fwd.h"

#include "flatten.h"

result_buf flatten_input(float* input,
                         layer_t* layers,
                         int lnum,
                         float* result) {
    // If the data to be flattened did not require any alignment padding, then
    // there's nothing we need to do.
    if (layers[lnum - 1].output_data_align_pad == 0)
        return input;

    // The "input" to im2row is the output of the previous layer.
    int input_rows = layers[lnum - 1].output_rows;
    int input_cols = layers[lnum - 1].output_cols;
    int input_height = layers[lnum - 1].output_height;
    int input_pad = layers[lnum - 1].output_data_align_pad;

    im2row(input, input_rows, input_cols, input_height, input_pad, result);

    return result;
}

void im2row(float* input,
            int input_rows,
            int input_cols,
            int input_height,
            int input_pad,
            float* result) {
    int img, hgt, row;

#if 0
    // Disable padding at the ends of row vectors for now. We need to revamp
    // the core layer_t structure to clearly distinguish between weight and
    // input dimensions for FC before we can do this.
    int output_pad = calc_padding(
            input_height * input_rows * input_cols, DATA_ALIGNMENT);
#endif

    ARRAY_4D(float, _input, input, input_height, input_rows,
             input_cols + input_pad);

    for (img = 0; img < NUM_TEST_CASES; img++) {
        for (hgt = 0; hgt < input_height; hgt++) {
            for (row = 0; row < input_rows; row++) {
                memcpy(result, &_input[img][hgt][row][0], input_cols * sizeof(float));
                result += input_cols;
            }
        }
#if 0
        memset((void*)result, 0, output_pad * sizeof(float));
        result += output_pad;
#endif
    }
}
