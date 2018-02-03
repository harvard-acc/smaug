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

#define INDEX_CONTAINER_TYPE int
#define INDEX_BITS (4)
// This many indices fit into the space of the containing type.
#define INDEX_PACKING_FACTOR (sizeof(INDEX_CONTAINER_TYPE) * 8 / INDEX_BITS)
#define DATA_TO_INDEX_RATIO (DATA_PACKING_FACTOR / INDEX_PACKING_FACTOR)

typedef struct _packed_csr_array_t {
    uint32_t* vals;
    uint32_t* col_idx;
    uint32_t* row_idx;
    size_t num_nonzeros;
    size_t num_rows;
    size_t total_buf_size;  // in bytes.
} packed_csr_array_t;

typedef struct _csr_array_t {
    float* vals;
    int* col_idx;
    int* row_idx;
    size_t num_nonzeros;
    size_t num_rows;
} csr_array_t;

csr_array_t compress_dense_data_csr(float* data, dims_t* data_dims);

void decompress_csr_data(csr_array_t* csr_data,
                         dims_t* data_dims,
                         float* dcmp_data);

void decompress_packed_csr_data(uint32_t* cmp_data,
                                uint32_t* cmp_col_idx,
                                uint32_t* cmp_row_idx,
                                dims_t* data_dims,
                                float* dcmp_data);

packed_csr_array_t pack_data_vec8_f16(csr_array_t csr_data,
                                      dims_t* data_dims);

csr_array_t alloc_csr_array_t(size_t num_nonzeros, size_t num_rows);
csr_array_t copy_csr_array_t(csr_array_t* existing_array);
void free_csr_array_t(csr_array_t* ptr);
void free_packed_csr_array_t(packed_csr_array_t* ptr);

#endif
