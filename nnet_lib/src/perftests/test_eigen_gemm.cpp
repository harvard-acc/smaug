#define TRANSPOSE_WEIGHTS 1

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "core/matrix_multiply.h"
#include "core/eigen/matrix_multiply.h"
#include "utility/init_data.h"
#include "utility/utility.h"
#include "utility/profiling.h"

#include "nnet_fwd.h"

int INPUT_DIM;
int NUM_TEST_CASES = 1;
int NUM_CLASSES;
float* sigmoid_table;

enum MODE {
  eigen,
  manual
};

void print_help() {
    printf("Usage: ./test_eigen_gemm mode\n"
           "  mode: eigen or manual\n");
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        print_help();
        return 1;
    }

    const char* mode_str = argv[1];
    MODE mode;
    if (strncmp(mode_str, "eigen", 5) == 0) {
        mode = eigen;
    } else if (strncmp(mode_str, "manual", 6) == 0) {
        mode = manual;
    } else {
        print_help();
        return 1;
    }

    int iterations = 256;
    int input_size = 4096;
    int weight_size = 4096 + 1;
    float* inputs = (float*)malloc_aligned(input_size * sizeof(float));
    float* weights = (float*)malloc_aligned(weight_size * input_size * sizeof(float));
    float* results = (float*)malloc_aligned((weight_size -1 ) * sizeof(float));

    if (mode == eigen) {
        for (int it = 0; it < iterations; it++) {
            nnet_eigen::matrix_multiply_with_bias_transpose(inputs,
                                        weights,
                                        NUM_TEST_CASES,
                                        input_size,
                                        weight_size,
                                        results);
        }
    } else {
        for (int it = 0; it < iterations; it++) {
            matrix_multiply_with_bias_transpose(inputs,
                                      weights,
                                      NUM_TEST_CASES,
                                      input_size,
                                      weight_size,
                                      results);
        }
    }

    free(inputs);
    free(weights);
    free(results);

    return 0;
}
