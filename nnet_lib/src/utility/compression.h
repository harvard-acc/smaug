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

// A CSR array that bitpacks its data together. See compression.c
// (pack_data_vec8_fp16) for a description of the packing methodology.
typedef struct _packed_csr_array_t {
    uint32_t* vals;
    uint32_t* col_idx;
    uint32_t* row_idx;
    size_t num_nonzeros;
    size_t num_rows;
    size_t total_buf_size;  // in bytes.
} packed_csr_array_t;

// A CSR array that stores data in unpacked 32-bit values. See compression.c
// (compress_dense_data_csr).
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

csr_array_t* compress_dense_data_csr(float* data, dims_t* data_dims);
void decompress_csr_data(csr_array_t* csr_data,
                         dims_t* data_dims,
                         float* dcmp_data);

void decompress_packed_csr_data(uint32_t* cmp_data,
                                uint32_t* cmp_col_idx,
                                uint32_t* cmp_row_idx,
                                dims_t* data_dims,
                                float* dcmp_data);

packed_csr_array_t* pack_data_vec8_f16(csr_array_t* csr_data,
                                       dims_t* data_dims);

csr_array_t* alloc_csr_array_t(size_t num_nonzeros, size_t num_rows);
csr_array_t* copy_csr_array_t(csr_array_t* existing_array);
void print_csr_array_t(csr_array_t* csr);
void free_csr_array_t(csr_array_t* ptr);
packed_csr_array_t* alloc_packed_csr_array_t(size_t num_total_vectors,
                                            size_t num_nonzeros,
                                            size_t num_rows);
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

csr_tile_list* compute_tiled_packed_csr_array_dims(packed_csr_array_t* csr,
                                                   int starting_row,
                                                   int num_rows,
                                                   int num_cols,
                                                   size_t max_tile_size);
csr_tile_list* tile_packed_csr_array_t(packed_csr_array_t* input,
                                       dims_t* dims,
                                       int starting_row,
                                       size_t max_tile_size);
void free_csr_tile_list(csr_tile_list* list);

#endif
