#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

#include "activation_functions.h"
#include "convolution.h"
#include "init_data.h"
#include "matrix_multiply.h"
#include "pooling.h"
#include "utility.h"
#include "zeropad.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

#ifdef GEM5_HARNESS
#include "gem5/aladdin_sys_connection.h"
#include "gem5/aladdin_sys_constants.h"
#endif

#include "nnet_fwd.h"

// Network layer configuration.
layer_type LAYER_TYPES[MAX_LAYERS] = {
    INPUT, CONV, POOL_MAX, FLATTEN, FC, FC, FC, OUTPUT
};
// Fully connected layer config.
int NUM_HIDDEN_UNITS[NUM_FC_LAYERS] = { 256, 256, 256 };

// Conv layer config: (kernel size, number of kernels)
int CONV_LAYERS[NUM_CONV_LAYERS][2] = { { 3, 2 } };
// Pool layer config: (pooling size, pooling stride)
int POOL_LAYERS[NUM_CONV_LAYERS][2] = { { 2, 2 } };

// Grab matrix n out of the doubly flattened w
// (w is a flattened collection of matrices, each flattened)
float* grab_matrix(float* w, int n, int* n_rows, int* n_columns) {
    int ind = 0;
    int i;
grab_matrix_loop:
    for (i = 0; i < n; i++) {
        ind += n_rows[i] * n_columns[i];
    }
    return w + ind;
}

#ifdef DMA_MODE
void grab_matrix_dma(float* weights,
                     int layer,
                     layer_t* layers) {
    size_t offset = 0;
    int i;
    // Start from layer idx 1 (to skip the input layer).
grab_matrix_dma_loop:
    for (i = 1; i < layer; i++) {
        offset += get_num_weights_layer(layers, i);
    }
    size_t size = get_num_weights_layer(layers, layer) * sizeof(float);
#if DEBUG == 1
    printf("dmaLoad weights, offset: %lu, size: %lu\n", offset*sizeof(float), size);
#endif
    if (size > 0)
        dmaLoad(weights, offset*sizeof(float), 0, size);
}
#endif

void print_debug(float* hid,
                 int rows_to_print,
                 int cols_to_print,
                 int num_columns) {
    int i, l;
    printf("\nHidden units:\n");
    for (i = 0; i < rows_to_print; i++) {
        for (l = 0; l < cols_to_print; l++) {
            printf("%f, ", hid[sub2ind(i, l, num_columns)]);
        }
        printf("\n");
    }
}

void print_debug4d(float* hid, int rows, int cols, int height) {
    int img, i, j, h;

    for (img = 0; img < NUM_TEST_CASES; img++) {
        printf("Input image: %d\n", img);
        for (h = 0; h < height; h++) {
            printf("Depth %d\n", h);
            for (i = 0; i < rows; i++) {
                for (j = 0; j < cols; j++) {
                    printf("%f, ",
                           hid[sub4ind(
                                   img, h, i, j, height, rows, cols)]);
                }
                printf("\n");
            }
        }
    }
}

// Dispatch to the appropriate activation function.
void activation_fun(float* hid, int size, float* sigmoid_table) {
    if (ACTIVATION_FUN == 0) {
        RELU(hid, size * NUM_TEST_CASES);
    } else if (ACTIVATION_FUN == 1) {
        sigmoid_lookup(hid, size * NUM_TEST_CASES, sigmoid_table);
    } else {
        sigmoidn(hid, size * NUM_TEST_CASES);
    }
}

bool run_layer(float* activations,
               float* weights,
               layer_t curr_layer,
               float* result_temp,
               float* sigmoid_table,
               bool do_activation_func) {
    bool result_in_input = false;
    layer_type l_type = curr_layer.type;
    if (l_type == FC) {
        PRINT_MSG("\nmatrix multiply with bias\n");
        MATRIX_MULTIPLY_WITH_BIAS(activations, weights, NUM_TEST_CASES,
                                  curr_layer.input_rows, curr_layer.input_cols,
                                  result_temp);
        PRINT_DEBUG4D(result_temp, curr_layer.output_rows,
                      curr_layer.output_cols, curr_layer.output_height);
    } else if (l_type == CONV) {
        PRINT_MSG("\nconvolution2d\n");
        convolution2d_zeropad(activations, weights, curr_layer, result_temp);
        PRINT_DEBUG4D(activations, curr_layer.output_rows, curr_layer.output_cols,
                      curr_layer.output_height);
        result_in_input = true;
    } else if (l_type == POOL_MAX) {
        PRINT_MSG("\nmax pooling\n");
        max_pooling(activations, result_temp, curr_layer);
        PRINT_DEBUG4D(result_temp, curr_layer.output_rows, curr_layer.output_cols,
                      curr_layer.output_height);
    } else if (l_type == FLATTEN) {
        // This is just a dummy layer. Return.
        result_in_input = true;
        do_activation_func = false;
    }

    if (do_activation_func) {
        PRINT_MSG("\nactivation function\n");
        // Pass through activation function
        if (result_in_input) {
            activation_fun(activations,
                           curr_layer.output_rows * curr_layer.output_cols *
                                   curr_layer.output_height,
                           sigmoid_table);

            PRINT_DEBUG4D(activations, curr_layer.output_rows,
                          curr_layer.output_cols, curr_layer.output_height);
        } else {
            activation_fun(result_temp,
                           curr_layer.output_rows * curr_layer.output_cols *
                                   curr_layer.output_height,
                           sigmoid_table);

            PRINT_DEBUG4D(result_temp, curr_layer.output_rows,
                          curr_layer.output_cols, curr_layer.output_height);
        }

    }
    return result_in_input;
}

// Does the forward predictive pass of a neural net.
//
// A float array of class predictions in row major format of size
// num_test_cases*num_labels will eventually be stored in either @hid or
// @hid_temp.
//
// A bool indicating where the final result is stored into the layers
// structure. If it is in @hid, then false, if in @hid_temp, true.
void nnet_fwd(float* hid,
              float* weights,
              layer_t* layers,
              int num_layers,
              float* hid_temp,
              float* sigmoid_table) {

    int i, j, l;
    layer_t curr_layer;

    // Alternate between reading from/writing to hid and hid_temp so we can
    // avoid copying matrices.
    bool result_in_temp = false;
    bool result_in_input = false;
    bool do_activation_func = true;

    if (PRINT_DATA_AND_WEIGHTS) {
        printf("DATA:\n");
        for (i = 0; i < NUM_TEST_CASES; i++) {
            printf("Datum %d:\n", i);
            for (j = 0; j < INPUT_DIM; j++) {
                printf("%e, ", hid[sub2ind(i, j, INPUT_DIM)]);
            }
            printf("\n");
        }
        printf("\nWEIGHTS:\n");
        for (i = 0; i < layers[1].input_rows; i++) {
            for (j = 0; j < layers[1].input_cols; j++) {
                printf("%f\n", weights[sub2ind(i, j, layers[1].input_cols)]);
            }
        }
        printf("\nEND WEIGHTS\n");
    }

    // FORMAT HERE IS H TIMES W, NOT W TIMES H!!!!!
    // SO EACH DATA POINT IS A ***ROW****

    l = 1;  // Skip the input layer.
#ifdef DMA_MODE
    dmaLoad(hid, 0, 0, NUM_TEST_CASES * INPUT_DIM * sizeof(float));
#endif

    //******************//
    //   PRIMARY LOOP   //
    //******************//

nnet_fwd_outer:
    for (l = 1; l < num_layers; l++) {
        curr_layer = layers[l];
        // Don't run the activation function on the last layer.
        do_activation_func = (l != num_layers - 1);

#ifdef DMA_MODE
        grab_matrix_dma(weights, l, layers);
#endif

        if (result_in_temp) {
            result_in_input = run_layer(hid_temp, weights, curr_layer, hid,
                                        sigmoid_table, do_activation_func);
        } else {
            result_in_input = run_layer(hid, weights, curr_layer, hid_temp,
                                        sigmoid_table, do_activation_func);
        }

        if (!result_in_input)
           result_in_temp = !result_in_temp;
    }

#ifdef DMA_MODE
    if (result_in_temp)
        dmaStore(hid_temp, 0, 0, NUM_TEST_CASES * NUM_CLASSES * sizeof(float));
    else
        dmaStore(hid, 0, 0, NUM_TEST_CASES * NUM_CLASSES * sizeof(float));
#endif

    layers[num_layers - 1].result_in_temp = (int)result_in_temp;

#ifdef DMA_MODE
    dmaStore(layers, 0, 0, num_layers*sizeof(layer_t));
#endif
}

size_t next_multiple(size_t request, size_t align) {
  size_t n = request/align;
  if (n == 0)
    return align;  // Return at least this many bytes.
  size_t remainder = request - n*align;
  if (remainder)
      return (n+1)*align;
  return request;
}

size_t calc_layer_intermediate_memory(layer_t layer) {
    size_t usage = 0;

    switch (layer.type) {
        case FC:
        case SOFTMAX:
            usage = layer.output_rows * layer.output_cols;
            break;
        case CONV:
        case POOL_MAX:
        case POOL_AVG:
            usage = max(
                    layer.input_rows * layer.input_cols * layer.input_height,
                    layer.output_rows * layer.output_cols *
                            layer.output_height);
            break;
        default:
            usage = 0;
            break;
    }
    return usage * NUM_TEST_CASES;
}

int configure_network(layer_t** layers_ptr) {
    int i, err;
    int last_conv_layer = 0, last_pool_layer = 0, last_fc_layer = 0;
    int next_input_width, next_input_height;
    int total_layers = 0;

    // I assume total layers is at most one pooling layer per conv layer, plus
    // all FC layers, plus 1 (output).
    err = posix_memalign(
            (void**)layers_ptr, CACHELINE_SIZE,
            next_multiple(sizeof(layer_t) * (MAX_LAYERS), CACHELINE_SIZE));
    ASSERT_MEMALIGN(layers_ptr, err);

    layer_t* layers = *layers_ptr;

    next_input_width = INPUT_X;
    next_input_height = INPUT_Y;
    for (i = 0; i < MAX_LAYERS; i ++) {
        layers[i].type = LAYER_TYPES[i];
        if (layers[i].type == INPUT) {
            assert(i == 0 && "Input layer must be the first layer!");
            layers[i].input_rows = INPUT_Y;
            layers[i].input_cols = INPUT_X;
            layers[i].input_height = INPUT_Z;

            if (LAYER_TYPES[i+1] == FC) {
                layers[i].output_rows = 1;
                layers[i].output_cols = INPUT_X * INPUT_Y;
            } else {
                layers[i].output_rows = INPUT_Y;
                layers[i].output_cols = INPUT_X;
            }
            layers[i].output_height = INPUT_Z;
        } else if (layers[i].type == CONV) {
            layers[i].c_kernel_size = CONV_LAYERS[last_conv_layer][0];
            layers[i].input_height = layers[i-1].output_height;
            layers[i].output_height = CONV_LAYERS[last_conv_layer][1];
            // Input rows/cols must include zero padding.
            layers[i].input_rows =
                    layers[i - 1].output_rows + layers[i].c_kernel_size - 1;
            layers[i].input_cols =
                    layers[i - 1].output_cols + layers[i].c_kernel_size - 1;
            layers[i].output_rows = layers[i - 1].output_rows;
            layers[i].output_cols = layers[i - 1].output_cols;
            last_conv_layer++;
        } else if (layers[i].type == POOL_MAX) {
            // Assume that the first layer will not be a pooling layer.
            layers[i].p_size = POOL_LAYERS[last_pool_layer][0];
            layers[i].p_stride = POOL_LAYERS[last_pool_layer][1];
            layers[i].input_rows = layers[i-1].output_rows;
            layers[i].input_cols = layers[i-1].output_cols;
            layers[i].input_height = layers[i - 1].output_height;
            layers[i].output_rows = ((layers[i].input_rows - layers[i].p_size) /
                                     layers[i].p_stride) + 1;
            layers[i].output_cols = ((layers[i].input_cols - layers[i].p_size) /
                                     layers[i].p_stride) + 1;
            layers[i].output_height = layers[i].input_height;
            last_pool_layer++;
        } else if (layers[i].type == FLATTEN) {
            layers[i].input_rows = layers[i-1].output_rows;
            layers[i].input_cols = layers[i-1].output_cols;
            layers[i].input_height = layers[i-1].output_height;
            layers[i].output_rows = 1;
            layers[i].output_cols = layers[i].input_rows *
                                    layers[i].input_cols *
                                    layers[i].input_height;
            layers[i].output_height = 1;
        } else if (layers[i].type == FC) {
            layers[i].input_rows = layers[i-1].output_cols + 1;
            layers[i].input_cols = NUM_HIDDEN_UNITS[last_fc_layer];
            layers[i].input_height = layers[i-1].output_height;

            // The next layer's input rows is the number of this layer's columns + 1.
            layers[i].output_cols = layers[i].input_cols;
            layers[i].output_rows = 1;
            layers[i].output_height = 1;
            last_fc_layer++;
        } else if (layers[i].type == OUTPUT) {
            // Assume that the last layer must be fully connected.
            layers[i].type = FC;
            layers[i].input_rows = layers[i-1].output_cols + 1;
            layers[i].input_cols = NUM_CLASSES;
            layers[i].input_height = layers[i-1].output_height;

            layers[i].output_cols = NUM_CLASSES;
            layers[i].output_rows = 1;
            layers[i].output_height = 1;

            // This is the last layer. Break.
            total_layers = ++i;
            break;
        } else if (layers[i].type == END) {
            total_layers = i;
            break;
        } else {
            printf("Layer %d is an unsupported type!\n", i);
            assert(false);
        }
    }

    // Helpfully print out the network topology. Skip the input layer.
    for (i = 1; i < total_layers; i++) {
        layer_type type = layers[i].type;
        printf("Layer %d: ", i);
        if (type == CONV) {
            printf("Convolutional\n");
            printf("  Input size: %d x %d\n", layers[i].input_rows,
                   layers[i].input_cols);
            printf("  Output size: %d x %d\n", layers[i].output_rows,
                   layers[i].output_cols);
            printf("  Kernel size: %d x %d\n", layers[i].c_kernel_size,
                   layers[i].c_kernel_size);
            printf("  Num kernels: %d\n", layers[i].output_height);
        } else if (type == FC) {
            printf("Fully connected\n");
            printf("  Weights: %d x %d\n", layers[i].input_rows,
                   layers[i].input_cols);
        } else if (type == POOL_MAX) {
            printf("Max pooling\n");
            printf("  Input size: %d x %d\n", layers[i].input_rows,
                   layers[i].input_cols);
            printf("  Output size: %d x %d\n", layers[i].output_rows,
                   layers[i].output_cols);
            printf("  Field size: %d\n", layers[i].p_size);
            printf("  Stride: %d\n", layers[i].p_stride);
            printf("  Height: %d\n", layers[i].output_height);
        } else if (type == SOFTMAX) {
            printf("Softmax\n");
        } else if (type == FLATTEN) {
            printf("Flatten\n");
            printf("  Input size: %d x %d\n", layers[i].input_rows,
                   layers[i].input_cols);
            printf("  Output size: %d x %d\n", layers[i].output_rows,
                   layers[i].output_cols);
        }
    }
    return total_layers;
}

// This is the thing that we want to be good at in hardware
int main(int argc, const char* argv[]) {
    int i, j, err;

    // set random seed (need to #include <time.h>)
    srand(1);

    layer_t* layers;
    int total_layers = configure_network(&layers);
    printf("Size of layer configuration: %lu\n", total_layers * sizeof(layer_t));

    data_init_mode RANDOM_DATA = FIXED;
    data_init_mode RANDOM_WEIGHTS = FIXED;

    // hid and hid_temp are the two primary buffers that will store the input
    // and output of each layer. They alternate in which one is input and which
    // is output. All input activations are initially loaded into hid. For this
    // reason, hid and hid_temp may not be the same size; hid must be large
    // enough to store the input activations, but this is not a concern for
    // hid_temp.
    float* hid;
    float* hid_temp;
    size_t data_size = NUM_TEST_CASES * INPUT_DIM;

    printf("Setting up arrays\n");
    // Get the dimensions of the biggest matrix that will ever come out of
    // run_layer.
    size_t hid_temp_size = 0;
    for (i = 1; i < total_layers; i++) {
        size_t curr_layer_usage = calc_layer_intermediate_memory(layers[i]);
        if (curr_layer_usage > hid_temp_size)
            hid_temp_size = curr_layer_usage;
    }
    printf("  Largest intermediate output size is %lu\n", hid_temp_size);
    err = posix_memalign(
            (void**)&hid_temp, CACHELINE_SIZE,
            next_multiple(hid_temp_size * sizeof(float), CACHELINE_SIZE));
    ASSERT_MEMALIGN(hid_temp, err);
    size_t hid_size = max(data_size, hid_temp_size);
    printf("  hid has %lu elements\n", hid_size);
    err = posix_memalign(
            (void**)&hid, CACHELINE_SIZE,
            next_multiple(hid_size * sizeof(float), CACHELINE_SIZE));
    ASSERT_MEMALIGN(hid, err);

    // Initialize weights, data, and labels.
    float* weights;
    int w_size = get_total_num_weights(layers, total_layers);
    err = posix_memalign((void**)&weights, CACHELINE_SIZE,
                         next_multiple(w_size * sizeof(float), CACHELINE_SIZE));
    ASSERT_MEMALIGN(weights, err);
    printf("Total weights: %d\n", w_size);
    init_weights(weights, layers, total_layers, RANDOM_WEIGHTS, TRANSPOSE_WEIGHTS);

    int* labels;
    size_t label_size = NUM_TEST_CASES;
    err = posix_memalign(
            (void**)&labels, CACHELINE_SIZE,
            next_multiple(label_size * sizeof(int), CACHELINE_SIZE));
    ASSERT_MEMALIGN(labels, err);

    init_data(hid, NUM_TEST_CASES, INPUT_DIM, RANDOM_DATA);
    init_labels(labels, NUM_TEST_CASES, RANDOM_DATA);

    // This file is not looked at by aladdin so malloc is fine.
    // If I do the old version then I get a memory overflow, because the
    // max stack size is not big enough for TIMIT stuff.

    // Build the sigmoid lookup table
    // May want to change this to be "non-centered"
    // to avoid (sigmoid_coarseness - 1.0)
    // so we can use bit shift in lookup function with fixed point precisions
    printf("Setting up sigmoid lookup table\n");
    int sigmoid_coarseness = 1 << LG_SIGMOID_COARSENESS;
    float sigmoid_table[sigmoid_coarseness];
    float sig_step = (float)(SIG_MAX - SIG_MIN) / (sigmoid_coarseness - 1.0);
    float x_sig = (float)SIG_MIN;
    for (i = 0; i < sigmoid_coarseness; i++) {
        sigmoid_table[i] = conv_float2fixed(1.0 / (1.0 + exp(-x_sig)));
        // printf("%f, %f\n", x_sig, sigmoid_table[i]);
        x_sig += sig_step;
    }

    fflush(stdout);

    // -------------------------------------------------------- //
    //     THIS IS THE FUNCTION BEING SIMULATED IN HARDWARE     //
    // -------------------------------------------------------- //
#ifdef GEM5_HARNESS
    mapArrayToAccelerator(
            INTEGRATION_TEST, "hid", hid, hid_size * sizeof(float));
    mapArrayToAccelerator(
            INTEGRATION_TEST, "hid_temp", hid_temp, hid_size * sizeof(float));
    mapArrayToAccelerator(
            INTEGRATION_TEST, "weights", weights, w_size * sizeof(float));
    mapArrayToAccelerator(
            INTEGRATION_TEST, "layers", layers, total_layers * sizeof(layer_t));
    invokeAcceleratorAndBlock(INTEGRATION_TEST);
#else
    // Run a forward pass through the neural net
    printf("Running forward pass\n");
    // The function being synthesized
    nnet_fwd(hid, weights, layers, total_layers, hid_temp, sigmoid_table);
#endif

    // Print the result, maybe not all the test_cases
    int num_to_print = 1;
    // don't try to print more test cases than there are
    num_to_print =
            num_to_print < NUM_TEST_CASES ? num_to_print : NUM_TEST_CASES;

    // Compute the classification error rate
    float* result = layers[total_layers-1].result_in_temp ? hid_temp : hid;
    int num_errors = 0;
    for (i = 0; i < NUM_TEST_CASES; i++) {
        if (arg_max(result + i * NUM_CLASSES, NUM_CLASSES, 1) != labels[i]) {
            num_errors = num_errors + 1;
        }
    }
    float error_fraction = ((float)num_errors) / ((float)NUM_TEST_CASES);
    printf("Fraction incorrect (over %d cases) = %f\n", NUM_TEST_CASES,
           error_fraction);

    // Print the output labels and soft outputs.
    FILE* output_labels = fopen("output_labels.out", "w");
    for (i = 0; i < NUM_TEST_CASES; i++) {
        int pred = arg_max(result + i * NUM_CLASSES, NUM_CLASSES, 1);
        fprintf(output_labels, "Test %d: %d\n  [", i, pred);
        for (j = 0; j < NUM_CLASSES; j++)
            fprintf(output_labels, "%f  ", result[sub2ind(i, j, NUM_CLASSES)]);
        fprintf(output_labels, "]\n");
    }
    fclose(output_labels);

    free(hid);
    free(hid_temp);
    free(weights);
    free(labels);
    free(layers);
}
