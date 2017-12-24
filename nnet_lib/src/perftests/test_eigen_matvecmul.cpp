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
#include "utility/eigen/init_data.h"

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
    network_t network;
    layer_t* layer = (layer_t*)malloc(sizeof(layer_t));
    layer->inputs.rows = NUM_TEST_CASES;
    layer->inputs.cols = 4096;
    layer->inputs.height = 1;
    layer->inputs.align_pad = 0;
    layer->weights.rows = layer->inputs.cols + 1;
    layer->weights.cols = 4096;
    layer->weights.height = 1;
    layer->weights.align_pad = 0;
    network.layers = layer;
    network.depth = 1;

    data_init_mode init_mode = RANDOM;

    int input_size = layer->inputs.cols * layer->inputs.rows;
    int weight_size = layer->weights.rows * layer->weights.cols;

    float* inputs = (float*)malloc_aligned(input_size * sizeof(float));
    float* weights =
            (float*)malloc_aligned(weight_size * sizeof(float));
    float* results =
            (float*)malloc_aligned((layer->weights.rows - 1) * sizeof(float));

    // Initialize some random data.
    if (mode == eigen) {
        std::cout << "Running Eigen.\n";
          nnet_eigen::init_data(inputs, &network, NUM_TEST_CASES, init_mode);
          nnet_eigen::init_fc_weights(
                  weights, 1, layer->weights.rows, layer->weights.cols, init_mode);
      } else {
          std::cout << "Running manual.\n";
          init_data(inputs, &network, NUM_TEST_CASES, init_mode);
          init_fc_weights(weights,
                          layer->weights.height,
                          layer->weights.rows,
                          layer->weights.cols,
                          layer->weights.align_pad,
                          init_mode,
                          TRANSPOSE_WEIGHTS);
    }

    auto start = std::chrono::high_resolution_clock::now();

    if (mode == eigen) {
        for (int it = 0; it < iterations; it++) {
            nnet_eigen::matrix_multiply_with_bias(inputs,
                                                  weights,
                                                  NUM_TEST_CASES,
                                                  layer->weights.rows,
                                                  layer->weights.cols,
                                                  results);
        }
    } else {
        for (int it = 0; it < iterations; it++) {
            if (TRANSPOSE_WEIGHTS) {
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
    if (layer->weights.rows <= 11) {
        // If it's not too large, then print the actual result.
        print_debug(results, 1, layer->weights.cols, layer->weights.cols);
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

    return 0;
}
