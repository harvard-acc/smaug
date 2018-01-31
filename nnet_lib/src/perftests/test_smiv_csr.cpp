#include <stdio.h>
#include <string.h>

#include "utility/compression.h"
#include "utility/utility.h"

int INPUT_DIM;
int NUM_CLASSES;
int NUM_TEST_CASES = 1;
float* sigmoid_table = NULL;
float* exp_table = NULL;
sigmoid_impl_t SIGMOID_IMPL;

int main(int args, const char* argv[]) {
    layer_t layer;
    layer.weights = { 5, 20, 1, 0 };
    size_t total_size = get_dims_size(&layer.weights) * sizeof(float);

    float* decomp_1 = (float*)malloc_aligned(total_size);
    float* decomp_2 = (float*)malloc_aligned(total_size);
    int num_nonzeros = 9;
    float csr_weights[num_nonzeros] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
    int csr_col_idx[num_nonzeros] = { 15, 0, 3, 1, 8, 15, 0, 3, 10 };
    int csr_row_idx[layer.weights.rows + 1] = { 0, 2, 5, 7, 8, 9 };
    csr_array_t csr = alloc_csr_array_t(num_nonzeros, layer.weights.rows);
    memcpy(csr.vals, &csr_weights[0], num_nonzeros * sizeof(float));
    memcpy(csr.col_idx, &csr_col_idx[0], num_nonzeros * sizeof(int));
    memcpy(csr.row_idx, &csr_row_idx[0], (csr.num_rows + 1) * sizeof(int));

    printf("Testing non-packed CSR decompression.\n");
    memset(decomp_2, 0, total_size);
    decompress_csr_data(&csr, &layer.weights, decomp_2);
    print_debug(decomp_2,
                layer.weights.rows,
                layer.weights.cols,
                layer.weights.cols);

    printf("Testing packed CSR decompression.\n");
    packed_csr_array_t packed = pack_data_vec8_f16(csr, &layer.weights);
    memset(decomp_1, 0, total_size);
    decompress_packed_csr_data(packed.vals,
                               packed.col_idx,
                               packed.row_idx,
                               &layer.weights,
                               decomp_1);
    print_debug(decomp_1,
                layer.weights.rows,
                layer.weights.cols,
                layer.weights.cols);

    printf("\n\nTesting non-packed CSR compression.\n");
    csr_array_t recomp = compress_dense_data_csr(decomp_2, &layer.weights);
    printf("Data: ");
    bool data_equal = true;
    for (unsigned i = 0; i < recomp.num_nonzeros; i++) {
        printf("%3.0f, ", recomp.vals[i]);
        data_equal &= (recomp.vals[i] == csr.vals[i]);
    }
    printf("%s\n", data_equal ? "EQUAL": "NOT EQUAL");

    printf("Column indices: ");
    bool col_equal = true;
    for (unsigned i = 0; i < recomp.num_nonzeros; i++) {
        printf("%d, ", recomp.col_idx[i]);
        col_equal &= (recomp.col_idx[i] == csr.col_idx[i]);
    }
    printf("%s\n", col_equal ? "EQUAL": "NOT EQUAL");
    printf("Row indices: ");
    bool row_equal = true;
    for (unsigned i = 0; i < recomp.num_nonzeros; i++) {
        printf("%d, ", recomp.row_idx[i]);
        row_equal &= (recomp.row_idx[i] == csr.row_idx[i]);
    }
    printf("%s\n", row_equal ? "EQUAL": "NOT EQUAL");

    free(decomp_1);
    free(decomp_2);
    free_csr_array_t(&csr);
    free_csr_array_t(&recomp);
    return 0;
}
