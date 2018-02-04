#include <stdint.h>

#include "core/nnet_fwd_defs.h"
#include "core/smiv/params.h"
#include "utility/compression.h"
#include "utility/utility.h"

ALWAYS_INLINE
void unpack_values_at_row_smiv(uint32_t* csr_data,
                               int cmp_col_offset,
                               int fetch_index_vec,
                               float values_buffer[VECTOR_SIZE * 2],
                               int index_buffer[VECTOR_SIZE * 2]) {
    v16ph_t* _cmp_values = (v16ph_t*)csr_data;

    // Extract and decompress the values.
    PRINT_MSG_V("  Fetching packed values from %d\n", fetch_index_vec);
    v16ph_t curr_values = _cmp_values[fetch_index_vec];

#ifdef __clang__
    v8ph_t values0_f16 = __builtin_shufflevector(
            curr_values, curr_values, 0, 1, 2, 3, 4, 5, 6, 7);
    v8ph_t values1_f16 = __builtin_shufflevector(
            curr_values, curr_values, 8, 9, 10, 11, 12, 13, 14, 15);
#else
    v8ph_t values0_f16 =
            (v8ph_t){ curr_values[0], curr_values[1], curr_values[2],
                      curr_values[3], curr_values[4], curr_values[5],
                      curr_values[6], curr_values[7] };
    v8ph_t values1_f16 =
            (v8ph_t){ curr_values[8],  curr_values[9],  curr_values[10],
                      curr_values[11], curr_values[12], curr_values[13],
                      curr_values[14], curr_values[15] };
#endif
    v8fp_t values0_f32 = _CVT_PH_PS_256(values0_f16);
    v8fp_t values1_f32 = _CVT_PH_PS_256(values1_f16);

    // Extract the 4-bit compressed indices.
    unsigned idx0 =
            csr_data[cmp_col_offset + fetch_index_vec * DATA_TO_INDEX_RATIO];
    unsigned idx1 = csr_data[cmp_col_offset +
                             fetch_index_vec * DATA_TO_INDEX_RATIO + 1];

    unpack_idx0:
    for (int j = 0; j < VECTOR_SIZE; j++) {
        values_buffer[j] = values0_f32[j];
        index_buffer[j] = (idx0 >> (j * INDEX_BITS)) & 0xf;
    }
    unpack_idx1:
    for (int j = 0; j < VECTOR_SIZE; j++) {
        values_buffer[j + VECTOR_SIZE] = values1_f32[j];
        index_buffer[j + VECTOR_SIZE] = (idx1 >> (j * INDEX_BITS)) & 0xf;
    }
}

// Decompress data stored in a packed variation of CSR. Written for SMIV.
//
// Because Aladdin cannot understand pointers to a location in the middle of a
// scratchpad, we have to dereference all memory with respect to the same base
// address. If we have distinct data stored in different parts of the
// scratchpad, we access it by adding an offset to the array index. This
// requires that all of the data be contiguous.
//
// Args:
//   cmp_data: The base address of the entire data buffer (including values and
//     indices).  It is assumed that the packed data elements start at index 0.
//   cmp_col_offset: The offset (at 4-byte granularity) into @packed_spad at
//     which the packed column indices start (e.g. packed_spad[cmp_col_offset]
//     is the first set of 8 indices).
//   cmp_row_offset: The offset (at 4-byte granularity) into @packed_spad at
//     which the row index pairs start.
//   dest_offset: The offset (at 4-byte granularity) into @dcmp_data at which
//      we'll start writing the decompressed data.
//   data_dims: The dimensions of the uncompressed data.
//   dcmp_data: The base of the destination scratchpad to store the
//     uncompressed data.
void decompress_packed_csr_data_smiv_fxp(uint32_t* cmp_data,
                                         int cmp_col_offset,
                                         int cmp_row_offset,
                                         int dest_offset,
                                         dims_t* data_dims,
                                         float* dcmp_data) {
    int data_rows = data_dims->rows;
    int data_cols = data_dims->cols;
    int data_pad = data_dims->align_pad;

    PRINT_MSG_V("==== DECOMPRESSING ==== \n");
    decompress_row:
    for (int row = 0; row < data_rows; row++) {
        // Unpack the row index pair.
        uint32_t packed_idx_size = cmp_data[cmp_row_offset + row];
        int curr_row_start_idx = (packed_idx_size >> 16) & 0xffff;
        int curr_row_size = packed_idx_size & 0xffff;
        PRINT_MSG_V("Row %d\n", row);
        PRINT_MSG_V("  Row start idx: %d\n", curr_row_start_idx);
        PRINT_MSG_V("  Row size: %d\n", curr_row_size);

        int col_idx = -1;
        int num_elems_remaining = curr_row_size;
        decompress_col:
        for (int col = 0; col < curr_row_size; col += DATA_PACKING_FACTOR) {
            float values_buffer[VECTOR_SIZE * 2];
            int index_buffer[VECTOR_SIZE * 2];
            unpack_values_at_row_smiv(
                    cmp_data,
                    cmp_col_offset,
                    curr_row_start_idx + (col / DATA_PACKING_FACTOR),
                    values_buffer,
                    index_buffer);
            for (int i = 0; i < (int)DATA_PACKING_FACTOR; i++) {
                PRINT_MSG_V("  values_buffer[%d] = %f, index_buffer[%d] = %d\n",
                            i, values_buffer[i], i, index_buffer[i]);
            }
            decompress_write:
            for (int val = 0; val < min2(num_elems_remaining, 16); val++) {
                float value = values_buffer[val];
                col_idx += index_buffer[val] + 1;
                ASSERT(col_idx < data_cols + data_pad &&
                       "Column index exceeds width of matrix!");
                dcmp_data[dest_offset +
                          sub2ind(row, col_idx, data_cols + data_pad)] = value;
                PRINT_MSG_V(
                        "  Storing _data[%d][%d] = %f\n", row, col_idx, value);
            }
            num_elems_remaining -= 16;
        }
    }
}
