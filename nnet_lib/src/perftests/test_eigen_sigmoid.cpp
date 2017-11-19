#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "core/activation_functions.h"
#include "core/eigen/activation_functions.h"
#include "utility/init_data.h"
#include "utility/utility.h"
#include "utility/profiling.h"

#include "nnet_fwd.h"

int INPUT_DIM;
int NUM_TEST_CASES;
int NUM_CLASSES;
float* sigmoid_table;

void print_help() {
    printf("Usage: ./test_eigen_sigmoid mode\n"
           "  mode: eigen or manual\n"
           "    eigen - Use Eigen implementation of sigmoid.\n"
           "    manual - Use manual implementation of sigmoid.\n");
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

    int iterations = 256 * 1024;
    size_t size = 1024;
    float* inputs = (float*)malloc_aligned(size * sizeof(float));
    float* results = (float*)malloc_aligned(size * sizeof(float));

    layer_t fake_layer;
    fake_layer.type = OUTPUT;

    if (use_eigen) {
        for (int it = 0; it < iterations; it++) {
            nnet_eigen::activation_fun(inputs, size, SIGMOID, nullptr, results);
        }

    } else {
        for (int it = 0; it < iterations; it++) {
            activation_fun(inputs, size, SIGMOID, nullptr);
        }
    }

    free(inputs);
    free(results);

    return 0;
}
