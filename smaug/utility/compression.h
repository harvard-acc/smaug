/**
 * \file compression.h
 * \brief Functions to implement CSR compression/decompression.
 *
 * CSR compression/decompression is not supported currently in SMAUG, but it
 * may be added in the future. The underlying implementations, including
 * support for Aladdin modeling, all already exist.
 */

#ifndef _UTILITY_COMPRESSION_H_
#define _UTILITY_COMPRESSION_H_

#include <stdint.h>

#include "core/nnet_fwd_defs.h"
#include "utility/fp16_utils.h"

// The number of unpacked elements in the vector.
#ifndef VECTOR_SIZE
#define VECTOR_SIZE (8)
#elif VECTOR_SIZE != 8
#error "Existing VECTOR_SIZE is incompatible with compression vector size!"
#endif

#define UNPACKED_ELEMENT_SIZE (sizeof(float))
#define PACKED_ELEMENT_SIZE (sizeof(short))
#define TOTAL_VECTOR_BYTES (VECTOR_SIZE * UNPACKED_ELEMENT_SIZE)
// This many packed data elements fit into the space of the original vector.
#define DATA_PACKING_FACTOR (VECTOR_SIZE * UNPACKED_ELEMENT_SIZE / PACKED_ELEMENT_SIZE)

typedef int IndexContainerType;
#define INDEX_BITS (4)
// This many indices fit into the space of the containing type.
#define INDEX_PACKING_FACTOR (sizeof(IndexContainerType) * 8 / INDEX_BITS)
// This is the ratio between the number of packed FP values that can fit in a
// **32-byte vector** to the number of packed indices that can fit in a
// **32-bit value**.
#define DATA_TO_INDEX_RATIO (DATA_PACKING_FACTOR / INDEX_PACKING_FACTOR)

/**
 * A CSR array that bitpacks its data together. See compression.c
 * (pack_data_vec8_fp16) for a description of the packing methodology.
 */
typedef struct _packed_csr_array_t {
    packed_fp16* vals;
    uint32_t* col_idx;
    uint32_t* row_idx;
    size_t num_nonzeros;
    size_t num_total_vectors;
    size_t num_rows;
    size_t total_buf_size;  // in bytes.
} packed_csr_array_t;

/**
 * A CSR array that stores data in unpacked 32-bit values. See compression.c
 * (compress_dense_data_csr).
 */
typedef struct _csr_array_t {
    float* vals;
    int* col_idx;
    int* row_idx;
    size_t num_nonzeros;
    size_t num_rows;
} csr_array_t;

ALWAYS_INLINE
static inline uint16_t get_row_idx(uint32_t packed_row_idx_size) {
    return (packed_row_idx_size >> 16) & 0xffff;
}

ALWAYS_INLINE
static inline uint16_t get_row_size(uint32_t packed_row_idx_size) {
    return packed_row_idx_size & 0xffff;
}

ALWAYS_INLINE
static inline int create_packed_row(uint16_t row_idx, uint16_t row_size) {
    return ((row_idx << 16) & 0xffff0000) | (row_size & 0xffff);
}

/**
 * Compress an uncompressed matrix into the modified CSR format.
 *
 * The modified CSR format is based on the CSC format used in Deep
 * Compression (Han et al):
 *   1. The nonzero values are stored linearly.
 *   2. Indices are the relative offsets from the previous value to the
 *      next nonzero value position. They are represented as 4-bit values,
 *      so if any two nonzero values are spaced 16 columns or more apart,
 *      a padding zero is inserted into the data array with offset 15.
 *   3. Row indices are stored in unmodified CSR format. The value is equal to
 *      the sum of the number of nonzero values and padding zeros in this row.
 *      There is no additional zero padding added at the end of the row.
 *   4. Native types are used - float for the data, int for column and row
 *      indices.
 */
csr_array_t* compress_dense_data_csr(float* data, dims_t* data_dims);

/** Decompress data in unpacked modified CSR format. */
void decompress_csr_data(csr_array_t* csr_data,
                         dims_t* data_dims,
                         float* dcmp_data);

/**
 * Directly decompress data stored in a packed variation of CSR.
 *
 * @param cmp_data Compressed data, packed in groups of 16 x FP16 elements.
 * @param cmp_col_idx Relative 4-bit indices that indicate the number of zeros
 *        before the next value in cmp_values in the same row.
 * @param cmp_row_idx Packed pair of values for each row in the matrix. The
 *        first 16 bits indicate the starting index of the ith row (by 256 bit
 *        granularity), and the second 16 bits indicate the number of nonzero
 *        values in this row.
 * @param data_dims The dimensions of the uncompressed data.
 * @param dcmp_data The base of the uncompressed data buffer.
 */
void decompress_packed_csr_data(packed_fp16* cmp_data,
                                uint32_t* cmp_col_idx,
                                uint32_t* cmp_row_idx,
                                dims_t* data_dims,
                                float* dcmp_data);

/**
 * Compress an array of single-precision FP values to half precision.
 *
 * This does not perform CSR compression; it only performs precision reduction.
 * There are no requirements on the size of the FP array. Two FP values are
 * packed into a 32-bit value, with the first occupying bits 0-15 and the
 * second from 16-31.
 *
 * Returns a pointer to a malloc'ed fp16array_t object, whose size is equal to the
 * minimum number of 32-bit unsigned values required to store the packed data.
 * To use an existing buffer, pass its pointer to the dest_buf argument;
 * otherwise, pass NULL, and it will be autmoatically allocated.
 *
 * TODO: Make dest_buf a fp16array_t* pointer instead, so we don't have to
 * worry about memory leaks when we replace an existing fp16array_t object.
 */
fp16array_t* pack_data_fp16(farray_t* sp_data, packed_fp16* dest_buf);

/**
 * Decompress an array of half-precision FP values to single precision.
 *
 * This requires that the number of packed elements be a multiple of 4, so that
 * it can use 128-bit F16C instructions to efficiently unpack 4 values at a
 * time.
 *
 * Returns a pointer to a malloc'ed farray_t object holding the decompressed
 * data.  To use an existing buffer, pass its pointer to the dest_buf argument;
 * otherwise, pass NULL, and it will be autmoatically allocated.
 */
farray_t* unpack_data_fp16x4(fp16array_t* hp_data, float* dest_buf);

/**
 * Pack data in the modified CSR format into a more compact storage format.
 *
 * The packed, quantized format looks like:
 *   1. Each value is compressed to 16 bit half precision floating point.
 *   2. 16 FP16 values are packed into 32-byte vectors.
 *   3. New rows always start on vector-aligned addresses; they cannot
 *      cross vector boundaries.
 *   4. 8 4-bit integer offsets are packed into 32-bit integers.
 *   5. Each row index is represented as a 32-bit packed pair of values.
 *      a. Bits 0-15: The number of elements in this row.
 *      b. Bits 16-31: The vector index in the data array that marks the
 *         beginning of this row.
 */
packed_csr_array_t* pack_csr_array_vec8_f16(csr_array_t* csr_data,
                                            dims_t* data_dims);

csr_array_t* alloc_csr_array_t(size_t num_nonzeros, size_t num_rows);
csr_array_t* copy_csr_array_t(csr_array_t* existing_array);
void print_csr_array_t(csr_array_t* csr);
void free_csr_array_t(csr_array_t* ptr);

/**
 * Allocate memory to store a packed CSR array.
 *
 * This struct needs to be accessed as a contiguous block of memory by an
 * accelerator, so we need to allocate the memory as such. The pointers in the
 * struct are simply referring to locations in the middle of the block.
 */
packed_csr_array_t* alloc_packed_csr_array_t(size_t num_total_vectors,
                                            size_t num_nonzeros,
                                            size_t num_rows);

/**
 * Copy an existing packed CSR array into a new array.
 *
 * This fully duplicates the data and all the pointers and metadata.
 */
packed_csr_array_t* copy_packed_csr_array_t(packed_csr_array_t* existing_array);
void free_packed_csr_array_t(packed_csr_array_t* ptr);

// CSR decompression tiling functions.

typedef struct _csr_tile {
    int start_row;
    int num_elems;
    int num_rows;
    int num_vectors;
    // This is the size of the compressed array, including space for indices.
    size_t total_bytes;
    // This is the size that will be taken up by the decompressed array.
    size_t eff_total_bytes;
    packed_csr_array_t* array;
    struct _csr_tile* next_tile;
} csr_tile;

typedef struct _csr_tile_list {
    csr_tile* head;
    size_t len;
} csr_tile_list;

/**
 * Determine how to tile a large CSR array in as few tiles as possible.
 *
 * This returns a list of CSR array tiles, which denote the dimensions of each
 * tile and the required storage for the array. The number of tiles is
 * determined by the sparsity of the original array. Tiles are returned as a
 * linked-list.
 *
 * Args:
 *   csr: The original CSR array.
 *   starting_row: Begin tiling from this row in the original array.
 *   num_rows: Tile up to this many rows from starting_row.
 *   num_cols: The total number of columns in the decompressed array.
 *   max_tile_size: The maximum compressed size of each tile.
 *
 * Returns:
 *   A linked list of CSR tile dimensions.
 */
csr_tile_list* compute_tiled_packed_csr_array_dims(packed_csr_array_t* csr,
                                                   int starting_row,
                                                   int num_rows,
                                                   int num_cols,
                                                   size_t max_tile_size);
/**
 * Tile a large CSR array.
 *
 * The result is a linked list of CSR tiles. Each tile contains metadata about
 * the tile parameters (size, number of nonzeros, etc) as well as a
 * packed_csr_array_t object that contains the data itself.
 *
 * Each tile's row indices start from 0, so if the original array looked like
 * this, and there were two tiles:
 *    Orig:
 *       Row 0: [data, columns]
 *       Row 1: [data, columns]
 *       Row 2: [data, columns]
 *       Row 3: [data, columns]
 *       Row indices: [0, 1, 2, 3]
 *    Tiled:
 *       Row 0: [data, columns]
 *       Row 1: [data, columns]
 *       Row indices: [0, 1]
 *       ----
 *       Row 0: [data, columns]
 *       Row 1: [data, columns]
 *       Row indices: [0, 1]
 *
 * @param input A pointer to the original, complete CSR array.
 * @param dims The dimensions of the decompressed array.
 * @param starting_row Begin tiling the array from this row.
 * @param max_tile_size The maximum size in bytes of each tile.
 */
csr_tile_list* tile_packed_csr_array_t(packed_csr_array_t* input,
                                       dims_t* dims,
                                       int starting_row,
                                       size_t max_tile_size);
void free_csr_tile_list(csr_tile_list* list);

#endif
