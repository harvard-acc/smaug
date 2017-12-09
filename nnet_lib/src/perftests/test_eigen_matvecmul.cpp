#define TRANSPOSE_WEIGHTS 1

#define EIGEN_RUNTIME_NO_MALLOC
#include "Eigen/Dense"

#ifndef EIGEN_VECTORIZE
#warning "Vectorization is disabled!"
#endif

#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdio.h>

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

enum MODE { eigen, manual };

void print_help() {
    printf("Usage: ./test_eigen_matvecmul mode\n"
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

    // This does many iterations of a big matrix-vector multiply.
    int iterations = 256;
    int input_size = 4096 * NUM_TEST_CASES;
    int weight_size = 4096 + 1;

    float* inputs = (float*)malloc_aligned(input_size * sizeof(float));
    float* weights =
            (float*)malloc_aligned(weight_size * input_size * sizeof(float));
    float* results = (float*)malloc_aligned((weight_size - 1) * sizeof(float));

    // Initialize some random data.
    for (int i = 0; i < input_size; i++)
        inputs[i] = randfloat() - 0.5;

    init_fc_weights(weights, 1, weight_size, input_size, 0, RANDOM, true);

    if (mode == eigen) {
      std::cout << "Running Eigen.\n";
    } else {
      std::cout << "Running manual.\n";
    }

    auto start = std::chrono::high_resolution_clock::now();

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

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << "s\n";

    // Print something about the result.
    float sum = 0;
    for (int i = 0; i < weight_size; i++)
        sum += results[i];
    printf("sum = %f\n", sum);

    free(inputs);
    free(weights);
    free(results);

    return 0;
}
