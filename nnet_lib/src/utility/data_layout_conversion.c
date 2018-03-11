// Functions to flatten a set of 3D input activations into a stack of row
// vectors.
//
// These will handle removing any additional zero-padding at the end of each
// image row (for data alignment) during the flattening and re-zeropadding the
// final row vector as appropriate.
//
// These functions are NOT intended to be traced by LLVM-Tracer!

#include <string.h>

#include "core/nnet_fwd_defs.h"
#include "utility/compression.h"
#include "utility/data_layout_conversion.h"
#include "utility/utility.h"

// Returns the number of elements required to store the flattened image.
//
// Args:
//   layers: The global array of layer descriptors, starting from zero.
//   lnum: The layer that expects a flattened input image. The input dimensions
//     of the flattening routine is the output dims of the previous layer.
int im2row_size(layer_t* layers, int lnum) {
    // The "input" to im2row is the output of the previous layer.
    int input_rows = layers[lnum - 1].outputs.rows;
    int input_cols = layers[lnum - 1].outputs.cols;
    int input_height = layers[lnum - 1].outputs.height;
    int input_pad = layers[lnum - 1].outputs.align_pad;
    // The "output" padding of im2row is the padding required for the input to
    // the current layer.
    int output_pad = layers[lnum].inputs.align_pad;

    return NUM_TEST_CASES *
           ((input_height * input_rows * (input_cols + input_pad)) +
            output_pad);
}

void im2row_fp16_impl(packed_fp16* input,
                      layer_t* layers,
                      int lnum,
                      packed_fp16* result) {
    int img, hgt, row;

    // The "input" to im2row is the output of the previous layer.
    int input_rows = layers[lnum - 1].outputs.rows;
    int input_cols = layers[lnum - 1].outputs.cols;
    int input_height = layers[lnum - 1].outputs.height;
    int input_pad = layers[lnum - 1].outputs.align_pad;
    // The "output" padding of im2row is the padding required for the input to
    // the current layer.
    int output_pad = layers[lnum].inputs.align_pad;

    ARRAY_4D(float16, _input, input, input_height, input_rows,
             input_cols + input_pad);
    ARRAY_1D(float16, _result, result);

    for (img = 0; img < NUM_TEST_CASES; img++) {
        for (hgt = 0; hgt < input_height; hgt++) {
            for (row = 0; row < input_rows; row++) {
                memcpy(_result, &_input[img][hgt][row][0],
                       input_cols * sizeof(float16));
                _result += input_cols;
            }
        }
        memset((void*)_result, 0, output_pad * sizeof(float16));
        _result += output_pad;
    }
}

void im2row_fp32_impl(float* input, layer_t* layers, int lnum, float* result) {
    int img, hgt, row;

    // The "input" to im2row is the output of the previous layer.
    int input_rows = layers[lnum - 1].outputs.rows;
    int input_cols = layers[lnum - 1].outputs.cols;
    int input_height = layers[lnum - 1].outputs.height;
    int input_pad = layers[lnum - 1].outputs.align_pad;
    // The "output" padding of im2row is the padding required for the input to
    // the current layer.
    int output_pad = layers[lnum].inputs.align_pad;

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

// Flatten the input image.
//
// This function only supports the Uncompressed data format.
//
// Args:
//   input: the data list object storing the input, which is assumed to be the
//      first entry in the list.
//   layers: The global layers descriptor.
//   lnum: The number of the layer that needs to have the outputs from the
//      previous layer flattened.
//   result: a data list object that **may** be used to store the flattened
//      output. If this function determines that nothing actually needs to be
//      done (this image can directly be reshaped into the flattened view),
//      this object is not used. Alternatively, if the flattened size exceeds
//      the size of the current results data list object, then it frees the
//      existing buffer and allocates new storage.
//
// Returns:
//   A data list object containing the flattened object. If no flattening needs
//   to be done, then this returns the @input data list; otherwise, it flattens
//   the input image and returns @result.
result_buf im2row(data_list* input,
                  layer_t* layers,
                  int lnum,
                  data_list* result) {
    // Check if we actually need to do anything.
    if (layers[lnum - 1].outputs.align_pad == 0 &&
        layers[lnum].inputs.align_pad == 0)
        return input;

    int result_size = im2row_size(layers, lnum);
    result = create_new_data_list_if_necessary(
            result, result_size, input->type[0]);
    if (result->type[0] == Uncompressed) {
        im2row_fp32_impl(input->data[0].dense->d, layers, lnum,
                         result->data[0].dense->d);
    } else {
        im2row_fp16_impl(input->data[0].dense_hp->d, layers, lnum,
                         result->data[0].dense_hp->d);
    }

    return result;
}

// NCHW -> NHWC

dims_t nchw_to_nhwc_dims(dims_t* input_dims, unsigned data_alignment) {
    dims_t nhwc = { input_dims->cols, input_dims->height, input_dims->rows,
                    calc_padding(input_dims->height, data_alignment) };
    return nhwc;
}

dims_t convert_nchw_to_nhwc(data_list* input,
                            int data_index,
                            int num_inputs,
                            dims_t input_dims,
                            unsigned data_alignment,
                            data_list* result) {
    data_storage_t type = input->type[data_index];
    result->type[data_index] = type;
    if (type == Uncompressed) {
        return convert_nchw_to_nhwc_farray(
                input->data[data_index].dense, num_inputs, input_dims,
                data_alignment, &result->data[data_index].dense);
    } else if (type == UncompressedHalfPrecision) {
        return convert_nchw_to_nhwc_fp16array(
                input->data[data_index].dense_hp, num_inputs, input_dims,
                data_alignment, &result->data[data_index].dense_hp);
    } else {
        fprintf(stderr,
                "[ERROR]: Cannot convert to NCHW from data storage type %s\n!",
                data_storage_str(type));
        assert(false &&
               "Invalid data storage type for data layout conversion!");
        return (dims_t){ 0, 0, 0, 0 };
    }
}

dims_t convert_nchw_to_nhwc_fp32(float* input,
                                 int num_inputs,
                                 dims_t input_dims,
                                 unsigned data_alignment,
                                 float** result) {
    const int input_channels = input_dims.height;
    const int input_rows = input_dims.rows;
    const int input_cols = input_dims.cols;
    const int input_pad = input_dims.align_pad;

    dims_t nhwc = nchw_to_nhwc_dims(&input_dims, data_alignment);
    if (*result == NULL) {
        const int size = num_inputs * get_dims_size(&nhwc);
        *result = (float*)malloc_aligned(size * sizeof(float));
    }

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

dims_t convert_nchw_to_nhwc_farray(farray_t* input,
                                   int num_inputs,
                                   dims_t input_dims,
                                   unsigned data_alignment,
                                   farray_t** result) {
    dims_t nhwc = nchw_to_nhwc_dims(&input_dims, data_alignment);
    *result = create_new_farray_if_necessary(
            *result, num_inputs * get_dims_size(&nhwc));
    convert_nchw_to_nhwc_fp32(
            input->d, num_inputs, input_dims, data_alignment, &(*result)->d);
    return nhwc;
}

dims_t convert_nchw_to_nhwc_fp16(packed_fp16* input,
                                 int num_inputs,
                                 dims_t input_dims,
                                 unsigned data_alignment,
                                 packed_fp16** result) {
    const int input_channels = input_dims.height;
    const int input_rows = input_dims.rows;
    const int input_cols = input_dims.cols;
    const int input_pad = input_dims.align_pad;

    dims_t nhwc = nchw_to_nhwc_dims(&input_dims, data_alignment);
    if (*result == NULL)
        *result = (packed_fp16*)malloc_aligned(
                num_inputs * get_dims_size(&nhwc) * sizeof(float16));

    // To simplify the code, internally index the data as 16-bit values.
    ARRAY_4D(float16, _input, input, input_channels, input_rows,
             input_cols + input_pad);
    ARRAY_4D(float16, _result, *result, nhwc.height, nhwc.rows,
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

dims_t convert_nchw_to_nhwc_fp16array(fp16array_t* input,
                                      int num_inputs,
                                      dims_t input_dims,
                                      unsigned data_alignment,
                                      fp16array_t** result) {
    dims_t nhwc = nchw_to_nhwc_dims(&input_dims, data_alignment);
    *result = create_new_fp16array_if_necessary(
            *result, num_inputs * get_dims_size(&nhwc));
    convert_nchw_to_nhwc_fp16(
            input->d, num_inputs, input_dims, data_alignment, &(*result)->d);
    return nhwc;
}

// NHWC -> NCHW

dims_t nhwc_to_nchw_dims(dims_t* input_dims, unsigned data_alignment) {
    dims_t nchw = { input_dims->height, input_dims->rows, input_dims->cols,
                    calc_padding(input_dims->rows, data_alignment) };
    return nchw;
}

dims_t convert_nhwc_to_nchw(data_list* input,
                            int data_index,
                            int num_inputs,
                            dims_t input_dims,
                            unsigned data_alignment,
                            data_list* result) {
    data_storage_t type = input->type[data_index];
    result->type[data_index] = type;
    if (type == Uncompressed) {
        return convert_nhwc_to_nchw_farray(
                input->data[data_index].dense, num_inputs, input_dims,
                data_alignment, &result->data[data_index].dense);
    } else if (type == UncompressedHalfPrecision) {
        return convert_nhwc_to_nchw_fp16array(
                input->data[data_index].dense_hp, num_inputs, input_dims,
                data_alignment, &result->data[data_index].dense_hp);
    } else {
        fprintf(stderr,
                "[ERROR]: Cannot convert to NHWC from data storage type %s\n!",
                data_storage_str(type));
        assert(false &&
               "Invalid data storage type for data layout conversion!");
        return (dims_t){ 0, 0, 0, 0 };
    }
}

dims_t convert_nhwc_to_nchw_fp32(float* input,
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

dims_t convert_nhwc_to_nchw_farray(farray_t* input,
                                   int num_inputs,
                                   dims_t input_dims,
                                   unsigned data_alignment,
                                   farray_t** result) {
    dims_t nchw = nhwc_to_nchw_dims(&input_dims, data_alignment);
    *result = create_new_farray_if_necessary(
            *result, num_inputs * get_dims_size(&nchw));
    convert_nhwc_to_nchw_fp32(
            input->d, num_inputs, input_dims, data_alignment, &(*result)->d);
    return nchw;
}

dims_t convert_nhwc_to_nchw_fp16(packed_fp16* input,
                                 int num_inputs,
                                 dims_t input_dims,
                                 unsigned data_alignment,
                                 packed_fp16** result) {
    dims_t nchw = nhwc_to_nchw_dims(&input_dims, data_alignment);
    if (*result == NULL) {
        *result = (packed_fp16*)malloc_aligned(
                num_inputs * get_dims_size(&nchw) * sizeof(float16));
    }

    // To simplify the code, internally index the data as 16-bit values.
    ARRAY_4D(float16, _input, input, input_dims.height, input_dims.rows,
             input_dims.cols + input_dims.align_pad);
    ARRAY_4D(float16, _result, *result, nchw.height, nchw.rows,
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

dims_t convert_nhwc_to_nchw_fp16array(fp16array_t* input,
                                      int num_inputs,
                                      dims_t input_dims,
                                      unsigned data_alignment,
                                      fp16array_t** result) {
    dims_t nchw = nhwc_to_nchw_dims(&input_dims, data_alignment);
    *result = create_new_fp16array_if_necessary(
            *result, num_inputs * get_dims_size(&nchw));
    convert_nhwc_to_nchw_fp16(
            input->d, num_inputs, input_dims, data_alignment, &(*result)->d);
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
int convert_nchw_to_blocked_nhwc(data_list* input,
                                 int data_index,
                                 int num_inputs,
                                 int block_size,
                                 dims_t input_dims,
                                 unsigned data_alignment,
                                 data_list* result) {
    data_storage_t type = input->type[data_index];
    result->type[data_index] = input->type[data_index];
    if (type == Uncompressed) {
        return convert_nchw_to_blocked_nhwc_fp32(
                input->data[data_index].dense, num_inputs, block_size,
                input_dims, data_alignment, &result->data[data_index].dense);
    } else if (type == UncompressedHalfPrecision) {
        return convert_nchw_to_blocked_nhwc_fp16(
                input->data[data_index].dense_hp, num_inputs, block_size,
                input_dims, data_alignment, &result->data[data_index].dense_hp);
    } else {
        fprintf(stderr,
                "[ERROR]: Cannot convert to NCHW from data storage type %s\n!",
                data_storage_str(type));
        assert(false &&
               "Invalid data storage type for data layout conversion!");
        return 0;
    }
}

int convert_nchw_to_blocked_nhwc_fp16(fp16array_t* input,
                                      int num_inputs,
                                      int block_size,
                                      dims_t input_dims,
                                      unsigned data_alignment,
                                      fp16array_t** result) {
    const int num_blocks = ceil(((float)input_dims.height) / block_size);
    const size_t total_converted_size =
            compute_blocked_nhwc_size(&input_dims, block_size, data_alignment);
    *result = create_new_fp16array_if_necessary(*result, total_converted_size);

    dims_t block_dims = input_dims;
    packed_fp16* curr_src = input->d;
    packed_fp16* curr_dst = (*result)->d;
    int channels_remaining = input_dims.height;
    while (channels_remaining > 0) {
        block_dims.height = min2(block_size, channels_remaining);
        dims_t nhwc = convert_nchw_to_nhwc_fp16(
                curr_src, num_inputs, block_dims, data_alignment, &curr_dst);
        curr_src += get_dims_size(&block_dims) / 2;  // packing factor is 2.
        curr_dst += get_dims_size(&nhwc) / 2;
        channels_remaining -= block_size;
    }
    return num_blocks;
}

int convert_nchw_to_blocked_nhwc_fp32(farray_t* input,
                                      int num_inputs,
                                      int block_size,
                                      dims_t input_dims,
                                      unsigned data_alignment,
                                      farray_t** result) {
    const int num_blocks = ceil(((float)input_dims.height) / block_size);
    const size_t total_converted_size =
            compute_blocked_nhwc_size(&input_dims, block_size, data_alignment);
    *result = create_new_farray_if_necessary(*result, total_converted_size);

    dims_t block_dims = input_dims;
    float* curr_src = input->d;
    float* curr_dst = (*result)->d;
    int channels_remaining = input_dims.height;
    while (channels_remaining > 0) {
        block_dims.height = min2(block_size, channels_remaining);
        dims_t nhwc = convert_nchw_to_nhwc_fp32(
                curr_src, num_inputs, block_dims, data_alignment, &curr_dst);
        curr_src += get_dims_size(&block_dims);
        curr_dst += get_dims_size(&nhwc);
        channels_remaining -= block_size;
    }
    return num_blocks;
}

// Convert blocked-channel NHWC format to NCHW.
//
// Args:
//   input: The input data list.
//   data_index: The index of the data in the data list.
//   num_inputs: Value of N.
//   block_size: Current block size.
//   output_dims: Input dimensions in NHWC format, where C is the total number
//     of channels.
//   data_alignment: The desired alignment for the innermost dim.
//   result: Pointer to output data list.
int convert_blocked_nhwc_to_nchw(data_list* input,
                                 int data_index,
                                 int num_inputs,
                                 int block_size,
                                 dims_t input_dims,
                                 unsigned data_alignment,
                                 data_list* result) {
    data_storage_t type = input->type[data_index];
    if (type == Uncompressed) {
        return convert_blocked_nhwc_to_nchw_fp32(
                input->data[data_index].dense, num_inputs, block_size,
                input_dims, data_alignment, &result->data[data_index].dense);
    } else if (type == UncompressedHalfPrecision) {
        return convert_blocked_nhwc_to_nchw_fp16(
                input->data[data_index].dense_hp, num_inputs, block_size,
                input_dims, data_alignment, &result->data[data_index].dense_hp);
    } else {
        fprintf(stderr,
                "[ERROR]: Cannot convert to NCHW from data storage type %s\n!",
                data_storage_str(type));
        assert(false &&
               "Invalid data storage type for data layout conversion!");
        return 0;
    }
}

int convert_blocked_nhwc_to_nchw_fp16(fp16array_t* input,
                                      int num_inputs,
                                      int block_size,
                                      dims_t input_dims,
                                      unsigned data_alignment,
                                      fp16array_t** result) {
    // Determine how large the final converted result will be.
    dims_t nchw = nhwc_to_nchw_dims(&input_dims, data_alignment);
    const size_t total_converted_size = get_dims_size(&nchw);
    *result = create_new_fp16array_if_necessary(*result, total_converted_size);

    const int num_blocks = ceil(((float)input_dims.cols) / block_size);
    dims_t block_dims = input_dims;
    packed_fp16* curr_src = input->d;
    packed_fp16* curr_dst = (*result)->d;
    int channels_remaining = input_dims.cols;
    while (channels_remaining > 0) {
        block_dims.cols = min2(block_size, channels_remaining);
        block_dims.align_pad = calc_padding(block_dims.cols, data_alignment);
        dims_t nchw = convert_nhwc_to_nchw_fp16(
                curr_src, num_inputs, block_dims, data_alignment, &curr_dst);
        curr_src += get_dims_size(&block_dims) / 2;
        curr_dst += get_dims_size(&nchw) / 2;
        channels_remaining -= block_size;
    }
    return num_blocks;
}

int convert_blocked_nhwc_to_nchw_fp32(farray_t* input,
                                      int num_inputs,
                                      int block_size,
                                      dims_t input_dims,
                                      unsigned data_alignment,
                                      farray_t** result) {
    // Determine how large the final converted result will be.
    dims_t nchw = nhwc_to_nchw_dims(&input_dims, data_alignment);
    const size_t total_converted_size = get_dims_size(&nchw);
    *result = create_new_farray_if_necessary(*result, total_converted_size);

    const int num_blocks = ceil(((float)input_dims.cols) / block_size);
    dims_t block_dims = input_dims;
    float* curr_src = input->d;
    float* curr_dst = (*result)->d;
    int channels_remaining = input_dims.cols;
    while (channels_remaining > 0) {
        block_dims.cols = min2(block_size, channels_remaining);
        block_dims.align_pad = calc_padding(block_dims.cols, data_alignment);
        dims_t nchw = convert_nhwc_to_nchw_fp32(
                curr_src, num_inputs, block_dims, data_alignment, &curr_dst);
        curr_src += get_dims_size(&block_dims);
        curr_dst += get_dims_size(&nchw);
        channels_remaining -= block_size;
    }
    return num_blocks;
}

void block_matrix_colwise_fp32(farray_t* input,
                               dims_t* input_dims,
                               int block_size,
                               int data_alignment,
                               data_list** result_ptr) {
    int num_blocks = FRAC_CEIL(input_dims->cols, block_size);
    int blocked_cols = min2(block_size, input_dims->cols);
    if (*result_ptr == NULL) {
        *result_ptr = init_data_list(num_blocks);
    }

    ARRAY_2D(float, _input, input->d, input_dims->cols + input_dims->align_pad);
    data_list* result = *result_ptr;
    int columns_remaining = input_dims->cols;
    for (int block = 0; block < num_blocks; block++) {
        int curr_block_cols = min2(columns_remaining, blocked_cols);
        int padding = calc_padding(curr_block_cols, data_alignment);
        int block_size = input_dims->rows * (curr_block_cols + padding);
        result->data[block].dense = create_new_farray_if_necessary(
                result->data[block].dense, block_size);
        result->type[block] = Uncompressed;

        ARRAY_2D(float, _result, result->data[block].dense->d,
                 curr_block_cols + padding);
        for (int r = 0; r < input_dims->rows; r++) {
            memcpy(&_result[r][0], &_input[r][block * blocked_cols],
                   curr_block_cols * sizeof(float));
        }
        columns_remaining -= curr_block_cols;
    }
}
