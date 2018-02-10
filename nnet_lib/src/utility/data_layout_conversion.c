// Functions to flatten a set of 3D input activations into a stack of row
// vectors.
//
// These will handle removing any additional zero-padding at the end of each
// image row (for data alignment) during the flattening and re-zeropadding the
// final row vector as appropriate.
//
// These functions are NOT intended to be traced by LLVM-Tracer!

#include <string.h>

#include "utility/data_layout_conversion.h"
#include "utility/utility.h"

void im2row_impl(float* input,
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
                memcpy(result, &_input[img][hgt][row][0],
                       input_cols * sizeof(float));
                result += input_cols;
            }
        }
        memset((void*)result, 0, output_pad * sizeof(float));
        result += output_pad;
    }
}

result_buf im2row(float* input, layer_t* layers, int lnum, float* result) {
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

    im2row_impl(input, input_rows, input_cols, input_height, input_pad,
                output_pad, result);

    return result;
}

// NCHW -> NHWC

dims_t nchw_to_nhwc_dims(dims_t* input_dims, unsigned data_alignment) {
    dims_t nhwc = { input_dims->cols, input_dims->height, input_dims->rows,
                    calc_padding(input_dims->height, data_alignment) };
    return nhwc;
}

dims_t convert_nchw_to_nhwc(float* input,
                            int num_inputs,
                            dims_t input_dims,
                            unsigned data_alignment,
                            float** result) {
    const int input_channels = input_dims.height;
    const int input_rows = input_dims.rows;
    const int input_cols = input_dims.cols;
    const int input_pad = input_dims.align_pad;

    dims_t nhwc = nchw_to_nhwc_dims(&input_dims, data_alignment);
    if (*result == NULL)
        *result = (float*)malloc_aligned(num_inputs * get_dims_size(&nhwc) *
                                         sizeof(float));
    ARRAY_4D(float, _input, input, input_channels, input_rows,
             input_cols + input_pad);
    ARRAY_4D(float, _result, *result, nhwc.height, nhwc.rows,
             nhwc.cols + nhwc.align_pad);
    for (int n = 0; n < num_inputs; n++) {
        for (int h = 0; h < input_rows; h++) {
            for (int w = 0; w < input_cols; w++) {
                for (int c = 0; c < input_channels + nhwc.align_pad; c++) {
                    if (c < input_channels)
                        _result[n][h][w][c] = _input[n][c][h][w];
                    else
                        _result[n][h][w][c] = 0;
                }
            }
        }
    }

    return nhwc;
}

// NHWC -> NCHW

dims_t nhwc_to_nchw_dims(dims_t* input_dims, unsigned data_alignment) {
    dims_t nchw = { input_dims->height, input_dims->rows, input_dims->cols,
                    calc_padding(input_dims->rows, data_alignment) };
    return nchw;
}

dims_t convert_nhwc_to_nchw(float* input,
                            int num_inputs,
                            dims_t input_dims,
                            unsigned data_alignment,
                            float** result) {
    dims_t nchw = nhwc_to_nchw_dims(&input_dims, data_alignment);
    if (*result == NULL) {
        *result = (float*)malloc_aligned(num_inputs * get_dims_size(&nchw) *
                                         sizeof(float));
    }
    ARRAY_4D(float, _input, input, input_dims.height, input_dims.rows,
             input_dims.cols + input_dims.align_pad);
    ARRAY_4D(float, _result, *result, nchw.height, nchw.rows,
             nchw.cols + nchw.align_pad);
    for (int n = 0; n < num_inputs; n++) {
        for (int c = 0; c < nchw.height; c++) {
            for (int h = 0; h < nchw.rows; h++) {
                for (int w = 0; w < nchw.cols + nchw.align_pad; w++) {
                    if (w < nchw.cols)
                        _result[n][c][h][w] = _input[n][h][w][c];
                    else
                        _result[n][c][h][w] = 0;
                }
            }
        }
    }

    return nchw;
}

// Compute the size in elements required store a block of NCHW data in blocked
// NHWC format.
size_t compute_blocked_nhwc_size(dims_t* input_dims,
                                int block_size,
                                int data_alignment) {
    // Determine how large the final converted result will be.
    const int num_blocks = ceil(((float)input_dims->height) / block_size);
    const int per_channel_size = input_dims->rows * input_dims->cols;
    const int last_block_size = input_dims->height % block_size;
    const int padded_block_size =
            block_size + calc_padding(block_size, data_alignment);
    return (num_blocks * padded_block_size + last_block_size) *
            per_channel_size;
}

// Convert NCHW to blocked-channel NHWC format.
//
// The result has five logical dimensions:
//   0: N
//   1: Block number
//   2: H
//   3: W
//   4: C within the block.
//
// Args:
//   input: The input data.
//   num_inputs: Value of N.
//   block_size: Desired block size.
//   input_dims: Input dimensions in NCHW format.
//   data_alignment: The desired alignment for the innermost dim.
//   result: Pointer to output data buffer. If NULL, this routine will malloc
//      aligned memory and update this pointer.
int convert_nchw_to_blocked_nhwc(float* input,
                                 int num_inputs,
                                 int block_size,
                                 dims_t input_dims,
                                 unsigned data_alignment,
                                 float** result) {
    const int num_blocks = ceil(((float)input_dims.height) / block_size);
    if (*result == NULL) {
        const size_t size = compute_blocked_nhwc_size(
                &input_dims, block_size, data_alignment);
        *result = (float*)malloc_aligned(size * sizeof(float));
    }

    dims_t block_dims = input_dims;
    float* curr_src = input;
    float* curr_dst = *result;
    int channels_remaining = input_dims.height;
    while (channels_remaining > 0) {
        block_dims.height = min2(block_size, channels_remaining);
        dims_t nhwc = convert_nchw_to_nhwc(curr_src,
                                           num_inputs,
                                           block_dims,
                                           data_alignment,
                                           &curr_dst);
        curr_src += get_dims_size(&block_dims);
        curr_dst += get_dims_size(&nhwc);
        channels_remaining -= block_size;
    }
    return num_blocks;
}

// Convert blocked-channel NHWC format to NCHW.
//
// Args:
//   input: The input data.
//   num_inputs: Value of N.
//   block_size: Current block size.
//   output_dims: Input dimensions in NHWC format, where C is the total number
//     of channels.
//   data_alignment: The desired alignment for the innermost dim.
//   result: Pointer to output data buffer. If NULL, this routine will malloc
//      aligned memory and update this pointer.
int convert_blocked_nhwc_to_nchw(float* input,
                                 int num_inputs,
                                 int block_size,
                                 dims_t input_dims,
                                 unsigned data_alignment,
                                 float** result) {
    if (*result == NULL) {
        // Determine how large the final converted result will be.
        dims_t nchw = nhwc_to_nchw_dims(&input_dims, data_alignment);
        const int total_converted_size = get_dims_size(&nchw);
        *result = (float*)malloc_aligned(total_converted_size * sizeof(float));
    }

    const int num_blocks = ceil(((float)input_dims.cols) / block_size);
    dims_t block_dims = input_dims;
    float* curr_src = input;
    float* curr_dst = *result;
    int channels_remaining = input_dims.cols;
    while (channels_remaining > 0) {
        block_dims.cols = min2(block_size, channels_remaining);
        block_dims.align_pad = calc_padding(block_dims.cols, data_alignment);
        dims_t nchw = convert_nhwc_to_nchw(curr_src,
                                           num_inputs,
                                           block_dims,
                                           data_alignment,
                                           &curr_dst);
        curr_src += get_dims_size(&block_dims);
        curr_dst += get_dims_size(&nchw);
        channels_remaining -= block_size;
    }
    return num_blocks;
}
