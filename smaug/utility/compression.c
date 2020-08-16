#include <stdint.h>
#include <string.h>

#include "core/nnet_fwd_defs.h"
#include "utility/compression.h"
#include "utility/utility.h"

#define MASK_AND_SHIFT(array, array_idx, vec_offset)                           \
    ((array)[array_idx] & 0xf) << (4 * (vec_offset))

// These two vectors must be of the same size.
typedef float v8fp_t __attribute__((__vector_size__(TOTAL_VECTOR_BYTES)));
typedef uint16_t v16short_t __attribute__((__vector_size__(TOTAL_VECTOR_BYTES)));
// This vector is used to manipulate the packed elements at VECTOR_SIZE
// granularity.
typedef uint16_t v8short_t
        __attribute__((__vector_size__(VECTOR_SIZE * PACKED_ELEMENT_SIZE)));

// These are used to pack and unpack half precision data.
typedef float v4fp_t __attribute__((__vector_size__(TOTAL_VECTOR_BYTES / 2)));
typedef uint16_t v4short_t
        __attribute__((__vector_size__(VECTOR_SIZE / 2 * PACKED_ELEMENT_SIZE)));

csr_array_t* alloc_csr_array_t(size_t num_nonzeros, size_t num_rows) {
    csr_array_t* csr = (csr_array_t*)malloc(sizeof(csr_array_t));
    csr->vals = (float*)malloc_aligned(num_nonzeros * sizeof(float));
    csr->col_idx = (int*)malloc_aligned(num_nonzeros * sizeof(int));
    csr->row_idx = (int*)malloc_aligned((num_rows + 1) * sizeof(int));
    csr->num_nonzeros = num_nonzeros;
    csr->num_rows = num_rows;
    return csr;
}

csr_array_t* copy_csr_array_t(csr_array_t* existing_array) {
    csr_array_t* csr = alloc_csr_array_t(
            existing_array->num_nonzeros, existing_array->num_rows);
    memcpy((void*)csr->vals, (void*)existing_array->vals,
           existing_array->num_nonzeros * sizeof(float));
    memcpy((void*)csr->col_idx, (void*)existing_array->col_idx,
           existing_array->num_nonzeros * sizeof(int));
    memcpy((void*)csr->row_idx, (void*)existing_array->row_idx,
           (existing_array->num_rows + 1) * sizeof(int));
    csr->num_nonzeros = existing_array->num_nonzeros;
    csr->num_rows = existing_array->num_rows;
    return csr;
}

void print_csr_array_t(csr_array_t* csr) {
    printf("Data: %lu elements\n", csr->num_nonzeros);
    for (unsigned i = 0; i < csr->num_nonzeros; i++) {
        printf("%2.8f, ", csr->vals[i]);
    }
    printf("\nColumn indices: ");
    for (unsigned i = 0; i < csr->num_nonzeros; i++) {
        printf("%d, ", csr->col_idx[i]);
    }
    printf("\nRow indices: ");
    for (unsigned i = 0; i < csr->num_rows + 1; i++) {
        printf("%d, ", csr->row_idx[i]);
    }
    printf("\n");
}

void free_csr_array_t(csr_array_t* ptr) {
    free(ptr->vals);
    free(ptr->col_idx);
    free(ptr->row_idx);
    free(ptr);
}

packed_csr_array_t* alloc_packed_csr_array_t(size_t num_total_vectors,
                                             size_t num_nonzeros,
                                             size_t num_rows) {
    packed_csr_array_t* csr =
            (packed_csr_array_t*)malloc(sizeof(packed_csr_array_t));
    size_t values_size = next_multiple(
            num_total_vectors * TOTAL_VECTOR_BYTES, TOTAL_VECTOR_BYTES);
    size_t col_idx_size = next_multiple(
            num_total_vectors * DATA_TO_INDEX_RATIO * sizeof(uint32_t),
            TOTAL_VECTOR_BYTES);
    size_t row_idx_size =
            next_multiple(num_rows * sizeof(uint32_t), TOTAL_VECTOR_BYTES);
    size_t total_buf_size = values_size + col_idx_size + row_idx_size;
    uint32_t* buffer = (uint32_t*)malloc_aligned(total_buf_size);
    csr->vals = buffer;
    csr->col_idx = csr->vals + values_size / sizeof(uint32_t);
    csr->row_idx = csr->col_idx + col_idx_size / sizeof(uint32_t);
    csr->num_nonzeros = num_nonzeros;
    csr->num_total_vectors = num_total_vectors;
    csr->num_rows = num_rows;
    csr->total_buf_size = total_buf_size;  // Used for setting TLB mappings.
    return csr;
}

packed_csr_array_t* copy_packed_csr_array_t(packed_csr_array_t* existing_array) {
    packed_csr_array_t* csr = alloc_packed_csr_array_t(
            existing_array->num_total_vectors, existing_array->num_nonzeros,
            existing_array->num_rows);
    memcpy((void*)csr->vals, (void*)existing_array->vals,
           existing_array->total_buf_size);
    return csr;
}

void free_packed_csr_array_t(packed_csr_array_t* ptr) {
    // There was only one memory allocation required for the entire struct.
    free(ptr->vals);
    free(ptr);
}

int compute_num_vectors_in_row(int num_elems_in_row) {
    return FRAC_CEIL(num_elems_in_row, DATA_PACKING_FACTOR);
}

fp16array_t* pack_data_fp16(farray_t* sp_data, packed_fp16* dest_buf) {
    fp16array_t* hp_data = (fp16array_t*)malloc(sizeof(fp16array_t));
    hp_data->size = (sp_data->size / 2) + (sp_data->size % 2);
    if (!dest_buf) {
        hp_data->d = (packed_fp16*)malloc_aligned(hp_data->size *
                                                  sizeof(packed_fp16));
        hp_data->freeable = true;
    } else {
        hp_data->d = dest_buf;
        hp_data->freeable = false;
    }
    memset(hp_data->d, 0,
           next_multiple(hp_data->size * sizeof(packed_fp16), CACHELINE_SIZE));
    for (size_t i = 0; i < sp_data->size; i++) {
        bool use_lo_half = (i % 2 == 0);
        hp_data->d[i / 2] |= ((int)_CVT_SS_SH(sp_data->d[i], 0))
                             << (use_lo_half ? 0 : 16);
    }
    return hp_data;
}

farray_t* unpack_data_fp16x4(fp16array_t* hp_data, float* dest_buf) {
    assert(hp_data->size % 4 == 0 &&
           "Half precision data size of must be a multiple of 4!");
    farray_t* sp_data = (farray_t*)malloc(sizeof(farray_t));
    sp_data->size = hp_data->size * 2;
    if (!dest_buf) {
        sp_data->d = (float*)malloc_aligned(sp_data->size * sizeof(float));
        sp_data->freeable = true;
    } else {
        sp_data->d = dest_buf;
        sp_data->freeable = false;
    }
    memset(sp_data->d, 0,
           next_multiple(sp_data->size * sizeof(float), CACHELINE_SIZE));

    for (size_t i = 0; i < hp_data->size / 4; i++) {
        v8short_t packed_data = *(v8short_t*)&hp_data->d[4 * i];
        v8short_t packed_data_hi =
                (v8short_t)_SHUFFLE_PD(packed_data, packed_data, 0x3);
        *((v4fp_t*)&sp_data->d[i * 8]) = (v4fp_t)_CVT_PH_PS_128(packed_data);
        *((v4fp_t*)&sp_data->d[i * 8 + 4]) =
                (v4fp_t)_CVT_PH_PS_128(packed_data_hi);
    }
    return sp_data;
}

csr_array_t* compress_dense_data_csr(float* data, dims_t* data_dims) {
    int num_values = get_dims_size(data_dims);
    // First we'll allocate space for the complete dense array; later, once
    // we've completely compressed the array, we'll copy it into a new smaller
    // sparse array. This is because due to the limited bitwidth for relative
    // offsets, we don't know how much internal zero padding is needed.
    csr_array_t* csr = alloc_csr_array_t(num_values, data_dims->rows);
    ARRAY_3D(float, _data, data, data_dims->rows, data_dims->cols);

    int num_nonzeros = 0;
    int curr_row_idx = 1;
    for (int h = 0; h < data_dims->height; h++) {
        for (int r = 0; r < data_dims->rows; r++) {
            PRINT_MSG_V("Row %d\n", r);
            // First, count the total number of nonzeros in this row.
            int num_elems_in_row = 0;
            int last_nz_idx = 0;
            for (int c = 0; c < data_dims->cols; c++) {
                if (_data[h][r][c] != 0) {
                    num_elems_in_row++;
                    last_nz_idx = c;
                }
            }
            PRINT_MSG_V("  Number of non zeros: %d, last idx: %d\n",
                        num_elems_in_row,
                        last_nz_idx);

            int next_offset = 0;
            for (int c = 0; c <= last_nz_idx; c++) {
                float curr_value = _data[h][r][c];
                if (curr_value == 0)
                    next_offset++;
                if (curr_value != 0 || next_offset == 16) {
                    if (next_offset == 16)
                        next_offset--;
                    csr->vals[num_nonzeros] = curr_value;
                    csr->col_idx[num_nonzeros] = next_offset;
                    PRINT_MSG_V(" Writing %5.5f, %d at index %d\n",
                                curr_value,
                                next_offset,
                                num_nonzeros);
                    num_nonzeros++;
                    next_offset = 0;
                }
            }
            csr->row_idx[curr_row_idx++] = num_nonzeros;
        }
    }
    csr->num_nonzeros = num_nonzeros;
    csr->row_idx[0] = 0;

    // Copy the data to a new sparse array and free the current one.
    csr_array_t* result = copy_csr_array_t(csr);
    free_csr_array_t(csr);
    return result;
}

packed_csr_array_t* pack_csr_array_vec8_f16(csr_array_t* csr_data,
                                            dims_t* data_dims) {
    PRINT_MSG_V("==== COMPRESSING ===== \n");
    // First, compute the overall size of the packed data, accounting for
    // row-alignment requirements.
    size_t total_num_vectors = 0;
    for (int row = 0; row < data_dims->rows; row++) {
        total_num_vectors += compute_num_vectors_in_row(
                csr_data->row_idx[row + 1] - csr_data->row_idx[row]);
    }
    PRINT_MSG_V("total num vectors: %lu\n", total_num_vectors);

    packed_csr_array_t* csr = alloc_packed_csr_array_t(
            total_num_vectors, csr_data->num_nonzeros, data_dims->rows);

    v16short_t* _data = (v16short_t*)csr->vals;
    // Independently track the current linear index into the compressed data
    // column, and row indices arrays.
    int curr_wgt_src_idx = 0;
    int curr_wgt_dst_idx = 0;
    int curr_col_src_idx = 0;
    int curr_col_dst_idx = 0;
    int curr_packed_row_idx = 0;
    // Track the number of elements we've packed, so we can sanity check to
    // make sure we never exceed num_nonzero.
    unsigned total_elements_packed = 0;
    for (int row = 0; row < data_dims->rows; row++) {
        int row_start_idx = csr_data->row_idx[row + 1];
        const int num_elems_in_row = row_start_idx - csr_data->row_idx[row];
        int num_packed_data_vectors =
                FRAC_CEIL(num_elems_in_row, VECTOR_SIZE * 2);
        PRINT_MSG_V("Row = %d\n", row);
        PRINT_MSG_V("  row start idx %d\n", row_start_idx);
        PRINT_MSG_V("  num elements in row %d\n", num_elems_in_row);
        PRINT_MSG_V("  num packed vectors %d\n", num_packed_data_vectors);
        int elems_remaining = num_elems_in_row;
        for (int vec = 0; vec < num_packed_data_vectors; vec++) {
            v8fp_t data_f32 = (v8fp_t){ 0 };
            v16short_t data_f16 = (v16short_t){ 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0, 0 };
            // We can only pack up to 8 SP floats at once, but the size of the
            // vector containing the packed data that we will eventually read
            // out is the same size in bytes as the uncompressed, so multiple
            // iterations are needed to thoroughly pack all possible elements
            // into the vector.
            for (int iter = 0;
                 iter < (int)(UNPACKED_ELEMENT_SIZE / PACKED_ELEMENT_SIZE);
                 iter++) {
                if (elems_remaining <= 0)
                    break;
                for (int col = 0; col < min2(elems_remaining, VECTOR_SIZE);
                     col++) {
                    data_f32[col] = csr_data->vals[curr_wgt_src_idx++];
                }
                v8short_t packed_f16 = _CVT_PS_PH_256(data_f32, 0);
                for (int i = 0; i < VECTOR_SIZE; i++) {
                    PRINT_MSG_V("  packed data: %#4x\n", packed_f16[i]);
                    data_f16[iter * VECTOR_SIZE + i] = packed_f16[i];
                }

                elems_remaining -= VECTOR_SIZE;
            }
            PRINT_MSG_V("  Storing to data[%d]\n", curr_wgt_dst_idx);
            _data[curr_wgt_dst_idx++] = data_f16;
        }

        // 4 bit indices -> 8 per 32-bit integer. They are indexed like so:
        // | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |
        // which means that if there are only two indices remaining in this
        // row, then they will be aligned towards the right 8 bits of the
        // 32-bit value.
        //
        // TODO: Column indices can straddle vector boundaries!!!! This is not
        // good!!!
        elems_remaining = num_elems_in_row;
        int num_packed_idx_vectors =
                num_packed_data_vectors * DATA_TO_INDEX_RATIO;
        for (int vec = 0; vec < num_packed_idx_vectors; vec++) {
            csr->col_idx[curr_col_dst_idx] = 0;
            for (int elem = 0;
                 elem < min2(elems_remaining, (int)INDEX_PACKING_FACTOR);
                 elem++) {
                csr->col_idx[curr_col_dst_idx] |= MASK_AND_SHIFT(
                        csr_data->col_idx, curr_col_src_idx++, elem);
            }
            PRINT_MSG_V("  packed col_idx[%d] = %#x\n",
                        curr_col_dst_idx,
                        csr->col_idx[curr_col_dst_idx]);
            elems_remaining -= INDEX_PACKING_FACTOR;
            curr_col_dst_idx++;
        }

        csr->row_idx[row] =
                create_packed_row(curr_packed_row_idx, num_elems_in_row);
        curr_packed_row_idx += num_packed_data_vectors;
        PRINT_MSG_V("  packed row = %#x\n", csr->row_idx[row]);
        total_elements_packed += num_elems_in_row;
    }
    assert(total_elements_packed == csr_data->num_nonzeros &&
           "The number of packed elements is not the same as the number of non "
           "zero elements specified!");

    PRINT_MSG_V("Compressed data:\n");
    for (unsigned i = 0; i < total_num_vectors; i++)
        PRINT_MSG_V("%#x ", csr->vals[i]);
    PRINT_MSG_V("\nCompressed col indices:\n");
    for (unsigned i = 0; i < total_num_vectors * DATA_TO_INDEX_RATIO; i++)
        PRINT_MSG_V("%#x ", csr->col_idx[i]);
    PRINT_MSG_V("\nCompressed row indices:\n");
    for (int i = 0; i < data_dims->rows; i++)
        PRINT_MSG_V("%#x ", csr->row_idx[i]);
    PRINT_MSG_V("\n");

    return csr;
}

void decompress_csr_data(csr_array_t* csr_data,
                         dims_t* data_dims,
                         float* dcmp_data) {
    int data_rows = data_dims->rows;
    int data_cols = data_dims->cols;
    int data_pad = data_dims->align_pad;

    ARRAY_2D(float, _data, dcmp_data, data_cols + data_pad);
    PRINT_MSG_V("==== DECOMPRESSING ==== \n");
    int curr_col_idx = 0;
    for (int row = 0; row < data_rows; row++) {
        int curr_row_start_idx = csr_data->row_idx[row];
        int next_row_start_idx = csr_data->row_idx[row + 1];
        int num_elems_in_row = next_row_start_idx - curr_row_start_idx;
        PRINT_MSG_V("Row %d\n", row);
        PRINT_MSG_V("  Row start idx: %d\n", curr_row_start_idx);
        PRINT_MSG_V("  Row size: %d\n", num_elems_in_row);

        // A column index of zero means there are no zeros in between it and
        // the previous nonzero value.  So, we need to implicitly add 1 to the
        // existing offset to get the new decompressed column index. This
        // boundary condition is easily handled with a starting offset of -1.
        int col_idx = 0;
        for (int col = 0; col < num_elems_in_row; col++) {
            col_idx += csr_data->col_idx[curr_col_idx];
            ASSERT(col_idx < (data_cols + data_pad) &&
                   "Column index exceeds width of matrix!");
            float value = csr_data->vals[curr_col_idx];
            _data[row][col_idx] = value;
            curr_col_idx++;
            col_idx++;
            PRINT_MSG_V("  Storing _data[%d][%d] = %f\n", row, col_idx, value);
        }
    }
}

/**
 * Unpack a vector's worth of CSR values and indices at a specific location.
 *
 * Since a vector stores 16 FP16 elements, this returns the unpacked
 * single-precision values and indices at that location in values_buffer and
 * index_buffer.
 *
 * @param cmp_values A pointer to the start of the packed CSR data.
 * @param cmp_col_idx A pointer to the start of the packed CSR column indices.
 * @param fetch_index_vec The index of the vector to fetch from the two arrays
 *                    above. This index refers to a VECTOR_ALIGNED memory
 *                    address.
 * @param values_buffer Stores up to 16 unpacked values.
 * @param index_buffer Stores up to 16 unpacked indices.
 */
void unpack_values_at_row(packed_fp16* cmp_values,
                          uint32_t* cmp_col_idx,
                          int fetch_index_vec,
                          float values_buffer[VECTOR_SIZE * 2],
                          int index_buffer[VECTOR_SIZE * 2]) {
    v16short_t* _cmp_values = (v16short_t*)cmp_values;

    // Extract and decompress the values.
    PRINT_MSG_V("  Fetching packed values from %d\n", fetch_index_vec);
    v16short_t curr_values = _cmp_values[fetch_index_vec];

#ifdef __clang__
    v8short_t values0_f16 = __builtin_shufflevector(
            curr_values, curr_values, 0, 1, 2, 3, 4, 5, 6, 7);
    v8short_t values1_f16 = __builtin_shufflevector(
            curr_values, curr_values, 8, 9, 10, 11, 12, 13, 14, 15);
#else
    v8short_t values0_f16 =
            (v8short_t){ curr_values[0], curr_values[1], curr_values[2],
                         curr_values[3], curr_values[4], curr_values[5],
                         curr_values[6], curr_values[7] };
    v8short_t values1_f16 =
            (v8short_t){ curr_values[8],  curr_values[9],  curr_values[10],
                         curr_values[11], curr_values[12], curr_values[13],
                         curr_values[14], curr_values[15] };
#endif
    v8fp_t values0_f32 = _CVT_PH_PS_256(values0_f16);
    v8fp_t values1_f32 = _CVT_PH_PS_256(values1_f16);

    // Extract the 4-bit compressed indices.
    unsigned idx0 = cmp_col_idx[fetch_index_vec * DATA_TO_INDEX_RATIO];
    unsigned idx1 = cmp_col_idx[fetch_index_vec * DATA_TO_INDEX_RATIO + 1];

    for (int j = 0; j < VECTOR_SIZE; j++) {
        values_buffer[j] = values0_f32[j];
        index_buffer[j] = (idx0 >> (j * INDEX_BITS)) & 0xf;
    }
    for (int j = 0; j < VECTOR_SIZE; j++) {
        values_buffer[j + VECTOR_SIZE] = values1_f32[j];
        index_buffer[j + VECTOR_SIZE] = (idx1 >> (j * INDEX_BITS)) & 0xf;
    }
}

void decompress_packed_csr_data(packed_fp16* cmp_data,
                                uint32_t* cmp_col_idx,
                                uint32_t* cmp_row_idx,
                                dims_t* data_dims,
                                float* dcmp_data) {
    int data_rows = data_dims->rows;
    int data_cols = data_dims->cols;
    int data_pad = data_dims->align_pad;

    ARRAY_2D(float, _data, dcmp_data, data_cols + data_pad);
    PRINT_MSG_V("==== DECOMPRESSING ==== \n");
    for (int row = 0; row < data_rows; row++) {
        // Row indices are themselves packed into an index and the number of
        // nonzeros in that row, 16 bits each. The index indicates where the
        // first element of this row is stored in the compressed data (as a
        // 32-byte vector index). We also need the number of nonzeros stored
        // separately in order to properly handle the fact that separate rows
        // cannot cross vector (32-byte) boundaries.
        uint32_t packed_idx_size = cmp_row_idx[row];
        int curr_row_start_idx = get_row_idx(packed_idx_size);
        int curr_row_size = get_row_size(packed_idx_size);
        PRINT_MSG_V("Row %d\n", row);
        PRINT_MSG_V("  Row start idx: %d\n", curr_row_start_idx);
        PRINT_MSG_V("  Row size: %d\n", curr_row_size);

        // A column index of zero means there are no zeros in between these two
        // nonzero values. We therefore need to implicitly add 1 to the
        // existing offset to figure out where to put the new value, and this
        // boundary condition is easily handled with a starting offset of -1.
        int col_idx = -1;
        int num_elems_remaining = curr_row_size;
        for (int col = 0; col < curr_row_size; col += DATA_PACKING_FACTOR) {
            float values_buffer[VECTOR_SIZE * 2];
            int index_buffer[VECTOR_SIZE * 2];
            unpack_values_at_row(
                    cmp_data,
                    cmp_col_idx,
                    curr_row_start_idx + (col / DATA_PACKING_FACTOR),
                    values_buffer,
                    index_buffer);
            for (int i = 0; i < (int)DATA_PACKING_FACTOR; i++) {
                PRINT_MSG_V("  values_buffer[%d] = %f, index_buffer[%d] = %d\n",
                            i, values_buffer[i], i, index_buffer[i]);
            }
            for (int val = 0; val < min2(num_elems_remaining, 16); val++) {
                float value = values_buffer[val];
                // Within each row, the column indices must be accumulated, as
                // they are relative positions, not absolute positions.
                col_idx += index_buffer[val] + 1;
                ASSERT(col_idx < data_cols + data_pad &&
                       "Column index exceeds width of matrix!");
                _data[row][col_idx] = value;
                PRINT_MSG_V(
                        "  Storing _data[%d][%d] = %f\n", row, col_idx, value);
            }
            num_elems_remaining -= 16;
        }
    }
}

//===--------------------------------------------==//
// Packed CSR array tiling functions.
//===--------------------------------------------==//

csr_tile* init_csr_tile() {
    csr_tile* tile = (csr_tile*)malloc(sizeof(csr_tile));
    tile->start_row = 0;
    tile->num_elems = 0;
    tile->num_rows = 0;
    tile->num_vectors = 0;
    tile->total_bytes = 0;
    tile->eff_total_bytes = 0;
    tile->next_tile = NULL;
    return tile;
}

void free_csr_tile_list(csr_tile_list* list) {
    if (!list || !list->head)
      return;
    csr_tile* head = list->head;
    do {
        csr_tile* next = head->next_tile;
        free_packed_csr_array_t(head->array);
        free(head);
        head = next;
    } while (head);
    free(list);
}

/**
 * Compute the required bytes to store this store.
 *
 * Args:
 *    num_elems_in_row: Number of nonzeros (plus padding zeros) in this row.
 *
 * Results:
 *    size_for_row: The total number of bytes required for this row, including
 *       all indices.
 *    num_vectors: The number of vectors required to store the packed data in
 *       this row.
 */
void compute_bytes_for_row_storage(int num_elems_in_row,
                                   size_t* size_for_row,
                                   int* num_vectors) {
    int num_data_vectors = compute_num_vectors_in_row(num_elems_in_row);
    int data_bytes = num_data_vectors * TOTAL_VECTOR_BYTES;
    int col_bytes = num_data_vectors * DATA_TO_INDEX_RATIO *
                    sizeof(IndexContainerType);
    int row_bytes = 1 * sizeof(int);
    *size_for_row = data_bytes + col_bytes + row_bytes;
    *num_vectors = num_data_vectors;
}

csr_tile_list* compute_tiled_packed_csr_array_dims(packed_csr_array_t* csr,
                                                   int starting_row,
                                                   int num_rows,
                                                   int num_cols,
                                                   size_t max_tile_size) {
    int curr_row = 0;
    size_t num_tiles = 1;
    csr_tile* head = init_csr_tile();
    csr_tile* curr = head;
    curr->start_row = 0;
    while (curr_row < num_rows) {
        int packed_row_size = csr->row_idx[starting_row + curr_row];
        int num_elems = get_row_size(packed_row_size);
        size_t size_for_row;
        int num_vectors;
        compute_bytes_for_row_storage(num_elems, &size_for_row, &num_vectors);
        if (curr->total_bytes + size_for_row > max_tile_size) {
            // Finish off the current tile.
            curr->total_bytes =
                    next_multiple(curr->total_bytes, TOTAL_VECTOR_BYTES);
            curr->eff_total_bytes =
                    next_multiple(curr->num_rows * num_cols * sizeof(float),
                                  TOTAL_VECTOR_BYTES);

            // Start the next tile. This code can be reused.
            curr->next_tile = init_csr_tile();
            curr = curr->next_tile;
            curr->start_row = curr_row;
            num_tiles++;
        }
        // Now update the current tile.
        curr->total_bytes += size_for_row;
        curr->num_rows++;
        curr->num_elems += num_elems;
        curr->num_vectors += num_vectors;
        curr_row++;
    }
    // Round the last tile size to vector alignment if necessary.
    curr->total_bytes = next_multiple(curr->total_bytes, TOTAL_VECTOR_BYTES);
    curr->eff_total_bytes = next_multiple(
         curr->num_rows * num_cols * sizeof(float), TOTAL_VECTOR_BYTES);

    csr_tile_list* list = (csr_tile_list*)malloc(sizeof(csr_tile_list));
    list->head = head;
    list->len = num_tiles;
    return list;
}

csr_tile_list* tile_packed_csr_array_t(packed_csr_array_t* input,
                                       dims_t* dims,
                                       int starting_row,
                                       size_t max_tile_size) {
    // TODO: Add a special case to handle the case when no tiling is needed.
    csr_tile_list* list = compute_tiled_packed_csr_array_dims(
            input, starting_row, dims->rows, dims->cols, max_tile_size);
    int start_row = starting_row;
    // Compute the starting offsets into the value and col_idx arrays, based on
    // the starting row.
    int packed_start_row_idx = get_row_idx(input->row_idx[start_row]);
    uint32_t value_offset =
            packed_start_row_idx * (TOTAL_VECTOR_BYTES / UNPACKED_ELEMENT_SIZE);
    uint32_t col_offset = packed_start_row_idx * DATA_TO_INDEX_RATIO;
    csr_tile* curr_tile = list->head;
    for (unsigned i = 0; i < list->len; i++) {
        curr_tile->array = alloc_packed_csr_array_t(curr_tile->num_vectors,
                                                    curr_tile->num_elems,
                                                    curr_tile->num_rows);
        // Now copy the relevant regions of the value and column index buffers.
        // TODO: Unify this behavior with alloc_packed_csr_array_t!!
        size_t value_bytes_to_copy =
                curr_tile->num_vectors * TOTAL_VECTOR_BYTES;
        size_t col_bytes_to_copy =
                (curr_tile->num_vectors * DATA_TO_INDEX_RATIO) *
                sizeof(IndexContainerType);

        memcpy((void*)curr_tile->array->vals, input->vals + value_offset,
               value_bytes_to_copy);
        memcpy((void*)curr_tile->array->col_idx, input->col_idx + col_offset,
               col_bytes_to_copy);
        // The row indices need to start from zero for each tile.
        int row_offset_reset = get_row_idx(input->row_idx[start_row]);
        for (int r = 0; r < curr_tile->num_rows; r++) {
            uint32_t packed_idx_size = input->row_idx[r + start_row];
            uint32_t row_offset = get_row_idx(packed_idx_size);
            uint32_t num_elems = get_row_size(packed_idx_size);
            curr_tile->array->row_idx[r] =
                    create_packed_row(row_offset - row_offset_reset, num_elems);
        }
        // The offset into the value array needs to be calculated with respect
        // to the type used in packed_csr_array_t (uint32_t*) while taking into
        // account the fact that values are packed into 32-byte vectors.
        value_offset +=
                curr_tile->num_vectors * TOTAL_VECTOR_BYTES / sizeof(uint32_t);
        // TODO: The offset into the value array is calculated as an 32-bit
        // integer offset but it should be calculated as a vector offset.
        col_offset += curr_tile->num_vectors * DATA_TO_INDEX_RATIO;

        start_row += curr_tile->num_rows;
        curr_tile = curr_tile->next_tile;
    }
    return list;
}
