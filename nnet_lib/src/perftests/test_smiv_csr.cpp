#include <math.h>
#include <stdio.h>
#include <string.h>

#include "core/smiv/smiv.h"
#include "utility/compression.h"
#include "utility/data_archive.h"
#include "utility/data_archive_bin.h"
#include "utility/data_archive_txt.h"
#include "utility/utility.h"

int INPUT_DIM;
int NUM_CLASSES;
int NUM_TEST_CASES = 1;
float* sigmoid_table = NULL;
float* exp_table = NULL;
sigmoid_impl_t SIGMOID_IMPL;

bool compare_iarrays(int* arr0, int* arr1, int size) {
    for (int i = 0; i < size; i++) {
        if (arr0[i] != arr1[i]) {
            printf("At %d: arr0 = %d, arr1 = %d\n", i, arr0[i], arr1[i]);
            return false;
        }
    }
    return true;
}

bool compare_farrays(float* arr0, float* arr1, int size) {
    for (int i = 0; i < size; i++) {
        if (arr0[i] != arr1[i]) {
            printf("At %d: arr0 = %f, arr1 = %f\n", i, arr0[i], arr1[i]);
            return false;
        }
    }
    return true;
}

bool compare_farrays_approx(float* arr0, float* arr1, int size) {
    for (int i = 0; i < size; i++) {
        float err = abs((arr0[i] - arr1[i]) / arr0[i]);
        if (err > 0.0001) {
            printf("At %d: arr0 = %f, arr1 = %f\n", i, arr0[i], arr1[i]);
            return false;
        }
    }
    return true;
}

void run_manual_test() {
    layer_t layer;
    const size_t num_rows = 5;
    layer.weights = (dims_t) { num_rows, 20, 1, 0 };
    size_t total_size = get_dims_size(&layer.weights) * sizeof(float);

    float* decomp_1 = (float*)malloc_aligned(total_size);
    float* decomp_2 = (float*)malloc_aligned(total_size);
    const int num_nonzeros = 9;
    float csr_weights[num_nonzeros] = { 0,   1.5, 2.5, 3.5, 4.5,
                                        5.5, 6.5, 7.5, 8.5 };
    int csr_col_idx[num_nonzeros] = { 15, 0, 3, 1, 8, 15, 0, 3, 10 };
    int csr_row_idx[num_rows + 1] = { 0, 2, 5, 7, 8, 9 };
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
        printf("%3.3f, ", recomp.vals[i]);
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
}

void run_file_test(const char* filename) {
    farray_t weights = { NULL, 0 };
    if (is_txt_file(filename)) {
        read_weights_from_txt_file(filename, &weights);
    } else {
        mmapped_file file = open_bin_data_file(filename);
        read_weights_from_bin_file(&file, &weights);
        close_bin_data_file(&file);
    }
    printf("Read %lu elements from file.\n", weights.size);

    printf("Testing non-packed CSR compression.\n");
    dims_t weights_dims = (dims_t){ 785, 256, 1, 0 };
    size_t layer_wgt_size = get_dims_size(&weights_dims);
    csr_array_t csr = compress_dense_data_csr(weights.d, &weights_dims);
    printf("Data: %lu elements\n", csr.num_nonzeros);
    for (unsigned i = 0; i < csr.num_nonzeros; i++) {
        printf("%3.3f, ", csr.vals[i]);
    }
    printf("\nColumn indices: ");
    for (unsigned i = 0; i < csr.num_nonzeros; i++) {
        printf("%d, ", csr.col_idx[i]);
    }
    printf("\nRow indices: ");
    for (unsigned i = 0; i < csr.num_rows + 1; i++) {
        printf("%d, ", csr.row_idx[i]);
    }
    printf("\n");

    float* decomp_1 = (float*)malloc_aligned(layer_wgt_size * sizeof(float));
    memset(decomp_1, 0, layer_wgt_size * sizeof(float));
    decompress_csr_data(&csr, &weights_dims, decomp_1);
    printf("Non-packed CSR decompression: %s\n",
           compare_farrays(decomp_1, weights.d, layer_wgt_size) ? "PASS"
                                                                : "FAIL");

    packed_csr_array_t packed = pack_data_vec8_f16(csr, &weights_dims);
    printf("Packed array fields: col offset = %ld, row offset = %ld\n",
           packed.col_idx - packed.vals, packed.row_idx - packed.vals);

    float* decomp_2 = (float*)malloc_aligned(layer_wgt_size * sizeof(float));
    memset(decomp_2, 0, layer_wgt_size * sizeof(float));
    decompress_packed_csr_data(packed.vals,
                               packed.col_idx,
                               packed.row_idx,
                               &weights_dims,
                               decomp_2);
    // We have to do an approximate comparison with the non-packed
    // decompression.
    printf("Reference packed CSR decompression: %s\n",
           compare_farrays_approx(decomp_1, decomp_2, layer_wgt_size) ? "PASS"
                                                                      : "FAIL");

    // Now use decomp_1 so we can compare the packed decompression exactly.
    memset(decomp_1, 0, layer_wgt_size * sizeof(float));
    decompress_packed_csr_data_smiv_fxp(packed.vals,
                                        packed.col_idx - packed.vals,
                                        packed.row_idx - packed.vals,
                                        &weights_dims,
                                        decomp_1);
    printf("SMIV packed CSR decompression: %s\n",
           compare_farrays(decomp_1, decomp_2, layer_wgt_size) ? "PASS"
                                                               : "FAIL");

    printf("Testing tiling.\n");
    layer_t layer;
    layer.weights = weights_dims;
    layer.wgt_storage_type = PackedCSR;
    layer.host_weights_buffer = (void*)&packed;
    csr_tile_list tile_list =
            tile_packed_csr_array_t(&packed, &layer, 128*1024);
    packed_csr_array_t* array = &tile_list.head->array;
    printf("Tiled packed array fields: col offset = %ld, row offset = %ld\n",
           array->col_idx - array->vals, array->row_idx - array->vals);
    bool pass = true;
    for (int i = 0; i < array->total_buf_size / sizeof(uint32_t); i++) {
        if (array->vals[i] != packed.vals[i]) {
            printf("At offset %d, array->vals = %#x, packed->vals = %#x\n", i,
                   array->vals[i], packed.vals[i]);
            pass = false;
        }
    }
    printf("Tiling: %s\n", pass ? "PASS": "FAIL");

    free(decomp_1);
    free(decomp_2);
    free_csr_tile_list(&tile_list);
    free_csr_array_t(&csr);
    free_packed_csr_array_t(&packed);
    free(weights.d);
}

int main(int argc, const char* argv[]) {
    if (argc == 1) {
        printf("Running manual test.\n\n");
        run_manual_test();
    } else if (argc == 2) {
        printf("Running file test with %s.\n\n", argv[1]);
        run_file_test(argv[1]);
    } else {
        printf("Usage:\n  ./test_smiv_csr <optional-data-filename>\n\n");
        printf("If no filename is given, this runs a hard-coded test. The "
               "filename should be pointing to a 784 x 256 x 256 x 256 x 10 "
               "FC network's pruned parameters.\n");
        return 1;
    }
    return 0;
}
