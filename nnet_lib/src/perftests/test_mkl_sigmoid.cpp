#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "arch/nnet_mkl.h"
#include "core/ref/activation_functions.h"
#include "core/ref/lookup_tables.h"
#include "core/mkl/activation_functions.h"
#include "utility/init_data.h"
#include "utility/utility.h"
#include "utility/profiling.h"

#include "nnet_fwd.h"

int INPUT_DIM;
int NUM_TEST_CASES = 1;
int NUM_CLASSES;
float* sigmoid_table;
float* exp_table;
sigmoid_impl_t SIGMOID_IMPL;

void print_help() {
    printf("Usage: ./test_mkl_sigmoid func mode\n"
           "  func: sigmoid, elu, tanh, selu\n"
           "  mode: mkl, lut, or manual\n"
           "    mkl - Use MKL implementation of sigmoid.\n"
           "    centered-lut - Use MKL centered LUT implementation of sigmoid.\n"
           "    noncentered-lut - Use MKL noncentered LUT implementation of sigmoid.\n"
           "    manual - Use manual implementation of sigmoid.\n");
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        print_help();
        return 1;
    }

    activation_type act_func;
    const char* func_str = argv[1];
    if (strncmp(func_str, "sigmoid", 8) == 0) {
        act_func = SIGMOID;
    } else if (strncmp(func_str, "tanh", 5) == 0) {
        act_func = TANH;
    } else if (strncmp(func_str, "elu", 4) == 0) {
        act_func = ELU;
    } else if (strncmp(func_str, "selu", 5) == 0) {
        act_func = SELU;
    } else {
        print_help();
        return 1;
    }

    const char* mode = argv[2];
    bool use_mkl = false;
    if (strncmp(mode, "mkl", 4) == 0) {
        use_mkl = true;
        SIGMOID_IMPL = ExpUnit;
    } else if (strncmp(mode, "centered-lut", 13) == 0) {
        use_mkl = true;
        SIGMOID_IMPL = CenteredLUT;
    } else if (strncmp(mode, "noncentered-lut", 16) == 0) {
        use_mkl = true;
        SIGMOID_IMPL = NoncenteredLUT;
    } else if (strncmp(mode, "manual", 7) == 0) {
        use_mkl = false;
        SIGMOID_IMPL = ExpUnit;
    } else {
        print_help();
        return 1;
    }

    init_sigmoid_table(&sigmoid_table);
    init_sigmoid_table(&exp_table);

    layer_t layer;
    int size = 1024;
    layer.inputs = { 1, 256, 1, 0 };
    layer.weights = { 256, size, 1, 0 };
    layer.outputs = { 1, size, 1, 0 };
    layer.type = FC;
    layer.activation = act_func;
    int iterations = 256 * size;
    float* inputs = (float*)malloc_aligned(size * sizeof(float));
    float* results = (float*)malloc_aligned(size * sizeof(float));

    device_t device;
    nnet_mkl::MklSession* session = new nnet_mkl::MklSession();
    device.session = session;

    // Initialize data.
    for (int i = 0; i < size; i++) {
        inputs[i] = ((float)i / size) - 0.5;
    }

    std::cout << "\n==========================\n";
    auto start = std::chrono::high_resolution_clock::now();

    if (use_mkl) {
        nnet_mkl::activation_fun(
                inputs, NUM_TEST_CASES, &layer, results, &device);
        for (int it = 0; it < iterations; it++) {
            session->run();
        }

    } else {
        for (int it = 0; it < iterations; it++) {
            activation_fun(inputs, NUM_TEST_CASES, size, act_func);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << "s\n\n";

    free(inputs);
    free(results);
    free(session);

    return 0;
}
