#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "arch/nnet_mkl.h"
#include "core/activation_functions.h"
#include "core/mkl/activation_functions.h"
#include "utility/init_data.h"
#include "utility/utility.h"
#include "utility/profiling.h"

#include "nnet_fwd.h"

int INPUT_DIM;
int NUM_TEST_CASES = 1;
int NUM_CLASSES;
float* sigmoid_table;

void print_help() {
    printf("Usage: ./test_mkl_sigmoid mode\n"
           "  mode: mkl or manual\n"
           "    mkl - Use MKL implementation of sigmoid.\n"
           "    manual - Use manual implementation of sigmoid.\n");
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        print_help();
        return 1;
    }

    const char* mode = argv[1];
    bool use_mkl = false;
    if (strncmp(mode, "mkl", 3) == 0) {
        use_mkl = true;
    } else if (strncmp(mode, "manual", 6) == 0) {
        use_mkl = false;
    } else {
        print_help();
        return 1;
    }

    int iterations = 256 * 1024;
    size_t size = 1024;
    float* inputs = (float*)malloc_aligned(size * sizeof(float));
    float* results = (float*)malloc_aligned(size * sizeof(float));

    device_t device;
    nnet_mkl::MklSession* session = new nnet_mkl::MklSession();
    device.session = session;

    // Initialize data.
    for (unsigned i = 0; i < size; i++) {
        inputs[i] = ((float)i / size) - 0.5;
    }

    std::cout << "\n==========================\n";
    auto start = std::chrono::high_resolution_clock::now();

    if (use_mkl) {
        nnet_mkl::activation_fun(
                inputs, NUM_TEST_CASES, size, SIGMOID, results, &device);
        for (int it = 0; it < iterations; it++) {
            session->run();
        }

    } else {
        for (int it = 0; it < iterations; it++) {
            activation_fun(inputs, NUM_TEST_CASES, size, SIGMOID, nullptr);
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
