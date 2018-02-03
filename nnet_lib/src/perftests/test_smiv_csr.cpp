#include <stdio.h>
#include <string.h>

#include "core/smiv/smiv.h"
#include "utility/compression.h"
#include "utility/utility.h"

int INPUT_DIM;
int NUM_CLASSES;
int NUM_TEST_CASES = 1;
float* sigmoid_table = NULL;
float* exp_table = NULL;
sigmoid_impl_t SIGMOID_IMPL;

bool compare_iarrays(int* arr0, int* arr1, int size) {
    for (int i = 0; i < size; i++)
        if (arr0[i] != arr1[i])
            return false;
    return true;
}

bool compare_farrays(float* arr0, float* arr1, int size) {
    for (int i = 0; i < size; i++)
        if (arr0[i] != arr1[i])
            return false;
    return true;
}

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
    memset(decomp_1, 0, total_size);
    decompress_csr_data(&csr, &layer.weights, decomp_1);
    print_debug(decomp_1,
                layer.weights.rows,
                layer.weights.cols,
                layer.weights.cols);

    printf("Testing reference packed CSR decompression.\n");
    packed_csr_array_t packed = pack_data_vec8_f16(csr, &layer.weights);
    memset(decomp_2, 0, total_size);
    decompress_packed_csr_data(packed.vals,
                               packed.col_idx,
                               packed.row_idx,
                               &layer.weights,
                               decomp_2);
    print_debug(decomp_2,
                layer.weights.rows,
                layer.weights.cols,
                layer.weights.cols);
    printf("Results are %s\n",
           compare_farrays(decomp_1, decomp_2, total_size / sizeof(float))
                   ? "EQUAL"
                   : "NOT EQUAL");
    free_packed_csr_array_t(&packed);

    printf("Testing SMIV packed CSR decompression.\n");
    packed = pack_data_vec8_f16(csr, &layer.weights);
    memset(decomp_2, 0, total_size);
    decompress_packed_csr_data_smiv_fxp(packed.vals,
                                        packed.col_idx - packed.vals,
                                        packed.row_idx - packed.vals,
                                        &layer.weights,
                                        decomp_2);
    print_debug(decomp_2,
                layer.weights.rows,
                layer.weights.cols,
                layer.weights.cols);
    printf("Results are %s\n",
           compare_farrays(decomp_1, decomp_2, total_size / sizeof(float))
                   ? "EQUAL"
                   : "NOT EQUAL");
    free_packed_csr_array_t(&packed);

    printf("\n\nTesting non-packed CSR compression.\n");
    csr_array_t recomp = compress_dense_data_csr(decomp_2, &layer.weights);
    printf("Data: ");
    for (unsigned i = 0; i < recomp.num_nonzeros; i++) {
        printf("%3.0f, ", recomp.vals[i]);
    }
    printf(": %s\n",
           compare_farrays(recomp.vals, csr.vals, recomp.num_nonzeros)
                   ? "EQUAL"
                   : "NOT EQUAL");

    printf("Column indices: ");
    for (unsigned i = 0; i < recomp.num_nonzeros; i++) {
        printf("%d, ", recomp.col_idx[i]);
    }
    printf(": %s\n",
           compare_iarrays(recomp.col_idx, csr.col_idx, recomp.num_nonzeros)
                   ? "EQUAL"
                   : "NOT EQUAL");

    printf("Row indices: ");
    for (unsigned i = 0; i < recomp.num_nonzeros; i++) {
        printf("%d, ", recomp.row_idx[i]);
    }
    printf(": %s\n",
           compare_iarrays(recomp.row_idx, csr.row_idx, recomp.num_rows)
                   ? "EQUAL"
                   : "NOT EQUAL");

    free(decomp_1);
    free(decomp_2);
    free_csr_array_t(&csr);
    free_csr_array_t(&recomp);
    return 0;
}
