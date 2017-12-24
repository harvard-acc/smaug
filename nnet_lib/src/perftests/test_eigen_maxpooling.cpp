#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "unsupported/Eigen/CXX11/Tensor"

#include "core/pooling.h"
#include "core/eigen/pooling.h"
#include "utility/init_data.h"
#include "utility/utility.h"
#include "utility/profiling.h"
#include "utility/eigen/init_data.h"
#include "utility/eigen/utility.h"

#include "nnet_fwd.h"

int INPUT_DIM;
int NUM_TEST_CASES = 2;
int NUM_CLASSES;
float* sigmoid_table;

enum MODE { eigen, manual };

void print_help() {
    printf("Usage: ./test_eigen_maxpooling mode\n"
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

    int iterations = 100;
    int in_rows = 224, in_cols = 224, chans = 3;
    int pool_size = 2;
    int pool_stride = 2;
    int out_rows = (in_rows - pool_size) / pool_stride + 1;
    int out_cols = (in_cols - pool_size) / pool_stride + 1;
    size_t total_in_size = in_rows * in_cols * chans * NUM_TEST_CASES;
    size_t total_out_size = out_rows * out_cols * chans * NUM_TEST_CASES;
    float* inputs = (float*)malloc_aligned(total_in_size * sizeof(float));
    float* results = (float*)malloc_aligned(total_out_size * sizeof(float));

    network_t network;
    layer_t layer;
    layer.type = POOLING;
    layer.pool = MAX;
    layer.inputs = {in_rows, in_cols, chans, 0};
    layer.outputs = {out_rows, out_cols, chans, 0};
    layer.weights = {pool_size, pool_size, 1, 0};
    layer.field_stride = pool_stride;
    network.layers = &layer;
    network.depth = 1;

    if (mode == eigen) {
        nnet_eigen::init_data(inputs, &network, NUM_TEST_CASES, FIXED);
    } else {
        init_data(inputs, &network, NUM_TEST_CASES, FIXED);
    }

    std::cout << "\n==========================\n";
    auto start = std::chrono::high_resolution_clock::now();

    if (mode == eigen) {
        for (int it = 0; it < iterations; it++) {
            nnet_eigen::max_pooling(inputs, results, layer);
        }

    } else {
        for (int it = 0; it < iterations; it++) {
            max_pooling(inputs, results, layer);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << "s\n\n";

    // Print something about the result.
    if (layer.outputs.rows <= 10) {
        // If it's not too large, then print the actual result.
        if (mode == eigen) {
            Eigen::TensorMap<Eigen::Tensor<float, 4>> map(
                    results, NUM_TEST_CASES, chans, out_rows, out_cols);
            nnet_eigen::print_debug4d(map);
        } else {
            print_debug4d(results, out_rows, out_cols, chans);
        }
    } else {
        // Otherwise, just print the sum of the result.
        float sum = 0;
        for (unsigned i = 0; i < total_out_size; i++)
            sum += results[i];
        printf("sum = %f\n", sum);
    }

    free(inputs);
    free(results);

    return 0;
}
