#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "arch/nnet_mkl.h"
#include "core/ref/convolution.h"
#include "core/ref/zeropad.h"
#include "core/mkl/convolution.h"
#include "utility/init_data.h"
#include "utility/utility.h"

#include "nnet_fwd.h"

int INPUT_DIM;
int NUM_TEST_CASES = 1; //2;
int NUM_CLASSES;
float* sigmoid_table;

void print_help() {
    printf("Usage: ./test_mkl_depthwise_conv mode\n"
           "  mode: mkl or manual\n"
           "    mkl - Use MKL implementation of depthwise convolution.\n"
           "    manual - Use manual implementation of depthwise "
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

    int iterations = 128;

    layer_t curr_layer;
    curr_layer.type = CONV_DEPTHWISE;
    curr_layer.inputs.height = 32;
    curr_layer.inputs.rows = 256;
    curr_layer.inputs.cols = 256;
    curr_layer.inputs.align_pad = 0;
    curr_layer.weights.height = 1;
    curr_layer.weights.rows = 3;
    curr_layer.weights.cols = 3;
    curr_layer.weights.align_pad = 0;
    curr_layer.field_stride = 1;
    curr_layer.c_padding = 1;
    curr_layer.outputs.height = curr_layer.inputs.height;
    curr_layer.outputs.rows = curr_layer.inputs.rows;
    curr_layer.outputs.cols = curr_layer.inputs.cols;
    curr_layer.outputs.align_pad = 0;
    curr_layer.inputs.rows += 2 * curr_layer.c_padding;
    curr_layer.inputs.cols += 2 * curr_layer.c_padding;

    device_t device;
    nnet_mkl::MklSession* session = new nnet_mkl::MklSession();
    device.session = session;

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
    if (!use_mkl) {
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

    if (use_mkl) {
        nnet_mkl::depthwise_convolution3d(
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
            convolution2d_depthwise_nopadding(
                    input_zeropad, weights, curr_layer, results);
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
    if (input_zeropad)
        free(input_zeropad);
    delete session;

    return 0;
}
