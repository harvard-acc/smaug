#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdio.h>

#include "mkldnn.hpp"

#include "core/matrix_multiply.h"
#include "core/mkl/matrix_multiply.h"
#include "utility/init_data.h"
#include "utility/utility.h"
#include "utility/profiling.h"

#include "nnet_fwd.h"

int INPUT_DIM;
int NUM_TEST_CASES = 5;
int NUM_CLASSES;
float* sigmoid_table;

enum MODE { mkl, manual };

void print_help() {
    printf("Usage: ./test_mkl_matvecmul mode\n"
           "  mode: mkl or manual\n");
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        print_help();
        return 1;
    }

    const char* mode_str = argv[1];
    MODE mode;
    if (strncmp(mode_str, "mkl", 3) == 0) {
        mode = mkl;
    } else if (strncmp(mode_str, "manual", 6) == 0) {
        mode = manual;
    } else {
        print_help();
        return 1;
    }
    bool use_mkl = (mode == mkl);

    // The transposed naive implementation runs much faster than the
    // non-transposed for obvious data layout reasons.
    bool use_transposed_gemm = true;

    // This does many iterations of a big matrix-vector multiply.
    int iterations = 256;
    network_t network;
    layer_t* layer = (layer_t*)malloc(sizeof(layer_t));
    layer->inputs.rows = 1;
    layer->inputs.cols = 4096;
    layer->inputs.height = 1;
    layer->inputs.align_pad = 0;
    layer->weights.rows = layer->inputs.cols + 1;
    layer->weights.cols = 4096;
    layer->weights.height = 1;
    layer->weights.align_pad = 0;
    layer->outputs.rows = layer->inputs.rows;
    layer->outputs.cols = layer->weights.cols;
    layer->outputs.height = 1;
    network.layers = layer;
    network.depth = 1;
    device_t device;
    nnet_mkl::MklSession* session;
    if (use_mkl) {
        session = new nnet_mkl::MklSession();
    } else {
        session = nullptr;
    }
    device.session = reinterpret_cast<void*>(session);

    data_init_mode init_mode = RANDOM;

    int input_size = layer->inputs.cols * layer->inputs.rows * NUM_TEST_CASES;
    int weight_size = layer->weights.rows * layer->weights.cols;

    float* inputs = (float*)malloc_aligned(input_size * sizeof(float));
    float* weights = (float*)malloc_aligned(weight_size * sizeof(float));
    float* results =
            (float*)malloc_aligned(layer->outputs.rows * layer->outputs.cols *
                                   NUM_TEST_CASES * sizeof(float));

    // Initialize some random data.
    init_data(inputs, &network, NUM_TEST_CASES, init_mode);
    init_fc_weights(weights,
                    layer->weights.height,
                    layer->weights.rows,
                    layer->weights.cols,
                    layer->weights.align_pad,
                    init_mode,
                    use_mkl ? true : use_transposed_gemm);

    std::cout << "MKL? " << use_mkl << "\n";
    auto start = std::chrono::high_resolution_clock::now();

    if (use_mkl) {
        nnet_mkl::matrix_multiply_with_bias(
                inputs, weights, layer, results, &device);
        for (int it = 0; it < iterations; it++) {
            session->run();
        }
    } else {
        for (int it = 0; it < iterations; it++) {
            if (use_transposed_gemm) {
                matrix_multiply_with_bias_transpose(inputs,
                                                    weights,
                                                    NUM_TEST_CASES,
                                                    layer->weights.rows,
                                                    layer->weights.cols,
                                                    results);
            } else {
                matrix_multiply_with_bias(inputs,
                                          weights,
                                          NUM_TEST_CASES,
                                          layer->weights.rows,
                                          layer->weights.cols,
                                          results);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << "s\n";

    // Print something about the result.
    if (layer->outputs.cols <= 11) {
        // If it's not too large, then print the actual result.
        print_debug(results,
                    NUM_TEST_CASES,
                    layer->outputs.cols,
                    layer->outputs.cols);
    } else {
        // Otherwise, just print the sum of the result.
        float sum = 0;
        for (int i = 0; i < layer->weights.rows - 1; i++)
            sum += results[i];
        printf("sum = %f\n", sum);
    }

    free(inputs);
    free(weights);
    free(results);
    free(layer);
    if (session)
        delete session;

    return 0;
}
