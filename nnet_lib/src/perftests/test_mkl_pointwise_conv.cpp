#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "arch/nnet_mkl.h"
#include "core/mkl/convolution.h"
#include "core/ref/convolution.h"
#include "core/ref/zeropad.h"
#include "utility/init_data.h"
#include "utility/utility.h"

#include "nnet_fwd.h"

int INPUT_DIM;
int NUM_TEST_CASES = 1; //2;
int NUM_CLASSES;
float* sigmoid_table;

void print_help() {
    printf("Usage: ./test_mkl_pointwise_conv mode\n"
           "  mode: mkl or manual\n"
           "    mkl - Use MKL implementation of pointwise convolution.\n"
           "    manual - Use manual implementation of pointwise "
           "convolution.\n");
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

    // The manual pointwise direct convolution is SUPER slow, so only one
    // iteration is required.
    int iterations = 1;
    int num_kernels = 256;

    layer_t curr_layer;
    curr_layer.type = CONV_POINTWISE;
    curr_layer.inputs.height = 1024;
    curr_layer.inputs.rows = 64;
    curr_layer.inputs.cols = 64;
    curr_layer.inputs.align_pad = 0;
    curr_layer.weights.height = 1;
    curr_layer.weights.rows = curr_layer.inputs.height + 1;
    curr_layer.weights.cols = num_kernels;
    curr_layer.weights.align_pad = 0;
    curr_layer.field_stride = 1;
    curr_layer.c_padding = 0;
    curr_layer.outputs.height = num_kernels;
    curr_layer.outputs.rows = curr_layer.inputs.rows;
    curr_layer.outputs.cols = curr_layer.inputs.cols;
    curr_layer.outputs.align_pad = 0;

    device_t device;
    nnet_mkl::MklSession* session = new nnet_mkl::MklSession();
    device.session = session;

    int total_input_size = get_input_activations_size(&curr_layer);
    int total_output_size = get_output_activations_size(&curr_layer);
    int total_weight_size = get_num_weights_layer(&curr_layer, 0);
    float* inputs = (float*)malloc_aligned(total_input_size * sizeof(float));
    float* weights = (float*)malloc_aligned(total_weight_size * sizeof(float));
    float* results = (float*)malloc_aligned(total_output_size * sizeof(float));
    assert(inputs && weights && results);

    // Initialize data.
    for (int i = 0; i < total_input_size; i++) {
        inputs[i] = gen_gaussian();
    }
    init_pointwise_conv_weights(weights,
                                curr_layer.weights.height,
                                curr_layer.weights.rows,
                                curr_layer.weights.cols,
                                curr_layer.weights.align_pad,
                                RANDOM,
                                false);

    PRINT_MSG("INPUTS\n");
    PRINT_DEBUG4D(inputs,
                  curr_layer.inputs.rows,
                  curr_layer.inputs.cols,
                  curr_layer.inputs.height);
    PRINT_MSG("WEIGHTS\n");
    PRINT_DEBUG4D(weights,
                  curr_layer.weights.rows,
                  curr_layer.weights.cols,
                  curr_layer.weights.height);

    if (use_mkl) {
        nnet_mkl::pointwise_convolution3d(
                inputs, weights, &curr_layer, results, &device);
    }
    std::cout << "\n==========================\n";

    auto start = std::chrono::high_resolution_clock::now();

    if (use_mkl) {
        for (int it = 0; it < iterations; it++) {
            session->run();
        }
    } else {
        for (int it = 0; it < iterations; it++) {
            convolution3d_pointwise_nopadding(
                    inputs, weights, curr_layer, results);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << "s\n\n";

    // Print something about the result.
    if (curr_layer.outputs.rows <= 10) {
        // If it's not too large, then print the actual result.
        print_debug4d(results,
                      curr_layer.outputs.rows,
                      curr_layer.outputs.cols,
                      curr_layer.outputs.height);
    } else {
        // Otherwise, just print the sum of the result.
        float sum = 0;
        for (int i = 0; i < total_output_size; i++)
            sum += results[i];
        printf("sum = %f\n", sum);
    }

    free(inputs);
    free(weights);
    free(results);
    delete session;

    return 0;
}
