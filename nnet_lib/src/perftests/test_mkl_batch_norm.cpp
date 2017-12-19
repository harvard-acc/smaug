#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "mkldnn.hpp"

#include "core/batch_norm.h"
#include "core/mkl/batch_norm.h"
#include "utility/init_data.h"
#include "utility/utility.h"
#include "utility/profiling.h"

#include "nnet_fwd.h"

int INPUT_DIM;
int NUM_TEST_CASES = 4;
int NUM_CLASSES;
float* sigmoid_table;

void print_help() {
    printf("Usage: ./test_mkl_batch_norm mode\n"
           "  mode: mkl or manual\n"
           "    mkl - Use MKL implementation of batch_norm.\n"
           "    manual - Use manual implementation of batch_norm.\n");
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        print_help();
        return 1;
    }

    const char* mode = argv[1];
    bool use_mkl = false;
    if (strncmp(mode, "mkl", 5) == 0) {
        use_mkl = true;
    } else if (strncmp(mode, "manual", 6) == 0) {
        use_mkl = false;
    } else {
        print_help();
        return 1;
    }

    printf("Using MKL: %d\n", use_mkl);

    int iterations = 1024;
    int batches = NUM_TEST_CASES;
    layer_t layer;
    layer.type = BATCH_NORM;  // Assume previous layer was FC.
    layer.inputs = { 64, 64, 1, 0 };
    layer.outputs = { 64, 64, 1, 0 };
    layer.weights = { 64 * 4, 64, 1, 0 };
    device_t device;
    nnet_mkl::MklSession* session = new nnet_mkl::MklSession();
    device.session = session;

    size_t input_size = get_input_activations_size(&layer);
    size_t weight_size = get_num_weights_layer(&layer, 0);
    size_t output_size = get_output_activations_size(&layer);
    float* inputs = (float*)malloc_aligned(input_size * sizeof(float));
    float* weights = (float*)malloc_aligned(weight_size * sizeof(float));
    float* results = (float*)malloc_aligned(output_size * sizeof(float));

    for (unsigned i = 0; i < input_size; i++) {
      inputs[i] = randfloat() - 0.5;
    }
    init_bn_weights(weights,
                    layer.weights.height,
                    layer.weights.rows,
                    layer.weights.cols,
                    layer.weights.align_pad,
                    RANDOM,
                    !use_mkl);

    auto start = std::chrono::high_resolution_clock::now();

    if (use_mkl) {
        nnet_mkl::batch_norm(
                inputs, weights, &layer, batches, results, &device);
        for (int it = 0; it < iterations; it++) {
            session->run();
        }

    } else {
        for (int it = 0; it < iterations; it++) {
            batch_norm_fxp(inputs, weights, &layer, batches, results);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << "s\n";

    // Use the result (just to be safe).
    float sum = 0;
    for (unsigned i = 0; i < input_size; i++)
      sum += results[i];
    printf("sum = %f\n", sum);

    free(inputs);
    free(weights);
    free(results);
    free(session);

    return 0;
}
