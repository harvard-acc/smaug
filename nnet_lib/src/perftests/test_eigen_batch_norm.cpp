#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>

#define EIGEN_RUNTIME_NO_MALLOC
#include "Eigen/Dense"

#ifndef EIGEN_VECTORIZE
#warning "Vectorization is disabled!"
#endif

#include "core/batch_norm.h"
#include "core/eigen/batch_norm.h"
#include "utility/init_data.h"
#include "utility/utility.h"
#include "utility/profiling.h"

#include "nnet_fwd.h"

int INPUT_DIM;
int NUM_TEST_CASES;
int NUM_CLASSES;
float* sigmoid_table;

void print_help() {
    printf("Usage: ./test_eigen_batch_norm mode\n"
           "  mode: eigen or manual\n"
           "    eigen - Use Eigen implementation of batch_norm.\n"
           "    manual - Use manual implementation of batch_norm.\n");
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        print_help();
        return 1;
    }

    const char* mode = argv[1];
    bool use_eigen = false;
    if (strncmp(mode, "eigen", 5) == 0) {
        use_eigen = true;
    } else if (strncmp(mode, "manual", 6) == 0) {
        use_eigen = false;
    } else {
        print_help();
        return 1;
    }

    printf("Using Eigen: %d\n", use_eigen);

    Eigen::internal::set_is_malloc_allowed(false);
    int iterations = 256 * 1024;
    int batches = 4;
    size_t input_size = 4096;
    float* inputs =
            (float*)malloc_aligned(input_size * batches * sizeof(float));
    float* weights =
            (float*)malloc_aligned(4 * input_size * sizeof(float));
    float* results =
            (float*)malloc_aligned(input_size * batches * sizeof(float));

    for (unsigned i = 0; i < input_size * batches; i++) {
      inputs[i] = randfloat() - 0.5;
    }
    init_bn_weights(weights, 1, 4, input_size, 0, RANDOM, false);

    layer_t fake_layer;
    fake_layer.type = OUTPUT;

    auto start = std::chrono::high_resolution_clock::now();

    if (use_eigen) {
        for (int it = 0; it < iterations; it++) {
            nnet_eigen::batch_norm(
                    inputs, weights, input_size, batches, results);
        }

    } else {
        for (int it = 0; it < iterations; it++) {
            batch_norm_fxp(inputs, weights, input_size, batches, results);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << "s\n";

    // Use the result (just to be safe).
    float sum = 0;
    for (int i = 0; i < input_size * batches; i++)
      sum += results[i];
    printf("sum = %f\n", sum);

    free(inputs);
    free(weights);
    free(results);

    return 0;
}
