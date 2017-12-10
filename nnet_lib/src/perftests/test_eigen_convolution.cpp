#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "Eigen/Dense"

#ifndef EIGEN_VECTORIZE
#warning "Vectorization is disabled!"
#endif

#include "core/convolution.h"
#include "core/eigen/convolution.h"
#include "core/zeropad.h"
#include "utility/init_data.h"
#include "utility/utility.h"

#include "nnet_fwd.h"

int INPUT_DIM;
int NUM_TEST_CASES = 2;
int NUM_CLASSES;
float* sigmoid_table;

void print_help() {
    printf("Usage: ./test_eigen_convolution mode\n"
           "  mode: eigen or manual\n"
           "    eigen - Use Eigen implementation of convolution.\n"
           "    manual - Use manual implementation of convolution.\n");
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

    int iterations = 128;

    layer_t curr_layer;
    curr_layer.type = CONV;
    curr_layer.inputs.height = 64;
    curr_layer.inputs.rows = 64;
    curr_layer.inputs.cols = 64;
    curr_layer.inputs.align_pad = 0;
    curr_layer.weights.height = curr_layer.inputs.height;
    curr_layer.weights.rows = 3;
    curr_layer.weights.cols = 3;
    curr_layer.weights.align_pad = 0;
    curr_layer.field_stride = 1;
    curr_layer.c_padding = 1;
    curr_layer.outputs.height = 2;
    curr_layer.outputs.rows = curr_layer.inputs.rows;
    curr_layer.outputs.cols = curr_layer.inputs.cols;
    curr_layer.outputs.align_pad = 0;

    int total_input_size = get_input_activations_size(&curr_layer);
    int total_output_size = get_output_activations_size(&curr_layer);
    int total_weight_size = get_num_weights_layer(&curr_layer, 0);
    float* inputs = (float*)malloc_aligned(total_input_size * sizeof(float));
    float* weights = (float*)malloc_aligned(total_weight_size * sizeof(float));
    float* results = (float*)malloc_aligned(total_output_size * sizeof(float));
    float* input_zeropad = NULL;
    assert(inputs && weights && results);

    // Initialize data.
    for (int i = 0; i < total_input_size; i++) {
        inputs[i] = i;
    }
    init_conv_weights(weights,
                      curr_layer.outputs.height,
                      curr_layer.weights.height,
                      curr_layer.weights.rows,
                      curr_layer.weights.cols,
                      curr_layer.weights.align_pad,
                      FIXED,
                      false);

    // For the manual implementation, we need to zeropad the input data first.
    if (!use_eigen) {
        curr_layer.inputs.rows += 2 * curr_layer.c_padding;
        curr_layer.inputs.cols += 2 * curr_layer.c_padding;
        int zeropadded_input_size = curr_layer.inputs.rows *
                                    curr_layer.inputs.cols *
                                    curr_layer.inputs.height * NUM_TEST_CASES;
        input_zeropad =
                (float*)malloc_aligned(zeropadded_input_size * sizeof(float));
        for (int img = 0; img < NUM_TEST_CASES; img++) {
            copy_zeropad_image3d(inputs,
                                 curr_layer.c_padding,
                                 img,
                                 curr_layer.outputs.rows,
                                 curr_layer.outputs.cols,
                                 curr_layer.inputs.height,
                                 curr_layer.outputs.align_pad,
                                 input_zeropad,
                                 curr_layer.inputs.rows,
                                 curr_layer.inputs.cols,
                                 curr_layer.inputs.align_pad);
        }
        PRINT_DEBUG4D(input_zeropad,
                      curr_layer.inputs.rows,
                      curr_layer.inputs.cols,
                      curr_layer.inputs.height);
    }

    std::cout << "\n==========================\n";

    auto start = std::chrono::high_resolution_clock::now();

    if (use_eigen) {
        for (int it = 0; it < iterations; it++) {
            nnet_eigen::convolution3d(inputs, weights, &curr_layer, results);
        }

    } else {
        for (int it = 0; it < iterations; it++) {
            convolution2d_no_padding(
                    input_zeropad, weights, curr_layer, results);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << "s\n\n";


    PRINT_DEBUG4D(results,
                  curr_layer.outputs.rows,
                  curr_layer.outputs.cols,
                  curr_layer.outputs.height);

    free(inputs);
    free(weights);
    free(results);
    if (input_zeropad)
        free(input_zeropad);

    return 0;
}
