#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "core/pooling.h"
#include "core/eigen/pooling.h"
#include "utility/init_data.h"
#include "utility/utility.h"
#include "utility/profiling.h"

#include "nnet_fwd.h"

int INPUT_DIM;
int NUM_TEST_CASES = 1;
int NUM_CLASSES;
float* sigmoid_table;

enum MODE {
  ROWMAJOR,
  COLMAJOR,
  MANUAL
};

void print_help() {
    printf("Usage: ./test_eigen_maxpooling mode\n"
           "  mode: rowmajor, colmajor, or manual\n");
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        print_help();
        return 1;
    }

    const char* mode_str = argv[1];
    MODE mode;
    if (strncmp(mode_str, "rowmajor", 8) == 0) {
        mode = ROWMAJOR;
    } else if (strncmp(mode_str, "colmajor", 8) == 0) {
        mode = COLMAJOR;
    } else if (strncmp(mode_str, "manual", 6) == 0) {
        mode = MANUAL;
    } else {
        print_help();
        return 1;
    }

    int iterations = 2; //256*8;
    int in_rows = 224, in_cols = 224, chans = 3;
    int pool_size = 2;
    int pool_stride = 2;
    int out_rows = (in_rows - pool_size) / pool_stride + 1;
    int out_cols = (in_cols - pool_size) / pool_stride + 1;
    size_t total_in_size = in_rows * in_cols * chans;
    size_t total_out_size = out_rows * out_cols * chans;
    float* inputs = (float*)malloc_aligned(total_in_size * sizeof(float));
    float* results = (float*)malloc_aligned(total_out_size * sizeof(float));

    layer_t fake_layer;
    fake_layer.type = POOLING;
    fake_layer.pool = MAX;
    fake_layer.inputs = {in_rows, in_cols, chans, 0};
    fake_layer.outputs = {out_rows, out_cols, chans, 0};
    fake_layer.weights = {pool_size, pool_size, 1, 0};
    fake_layer.field_stride = pool_stride;

    if (mode == ROWMAJOR) {
        for (int it = 0; it < iterations; it++) {
            nnet_eigen::max_pooling_rowmajor(inputs, results, fake_layer);
        }

    } else if (mode == COLMAJOR) {
        for (int it = 0; it < iterations; it++) {
            nnet_eigen::max_pooling_colmajor(inputs, results, fake_layer);
        }
    } else {
        for (int it = 0; it < iterations; it++) {
            max_pooling_image3d(inputs, 0, results, fake_layer);
        }
    }

    free(inputs);
    free(results);

    return 0;
}
