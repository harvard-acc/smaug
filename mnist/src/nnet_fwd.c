#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

#include "activation_functions.h"
#include "init_data.h"
#include "utility.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

#ifdef GEM5_HARNESS
#include "gem5/aladdin_sys_connection.h"
#include "gem5/aladdin_sys_constants.h"
#endif

#include "nnet_fwd.h"

// All the memory used in nnet:
// name           | type  | size/value
// ---------------|-------|--------------
// data           | float | NUM_TEST_CASES*INPUT_DIM
// weights        | float | INPUT_DIM * NUM_UNITS_1 +
//                |       | NUM_UNITS_1 * NUM_UNITS_2 +
//                |       | NUM_UNITS_2 * NUM_CLASSES
// num_test_cases | int   | NUM_TEST_CASES
// num_layers     | int   | NUM_FC_LAYERS
// num_units      | int   | NUM_FC_LAYERS + 2
// activation_fun | int   | ACTIVATION_FUN
// num_rows       | int   | NUM_FC_LAYERS + 1
// num_colums     | int   | NUM_FC_LAYERS + 1
// hid            | float | NUM_TEST_CASES * BIGGEST_ROW
// hid_temp       | float | NUM_TEST_CASES * BIGGEST_ROW

// Network layer configuration.
layer_type LAYER_TYPES[MAX_LAYERS] = {
    INPUT, FC, FC, FC, OUTPUT
};
// Fully connected layer config.
int NUM_HIDDEN_UNITS[NUM_FC_LAYERS] = { 256, 256, 256 };

/*
// Conv layer config: (kernel size, number of kernels)
int CONV_LAYERS[NUM_CONV_LAYERS][2] = { { 3, 1 } };
// Pool layer config: (pooling size, pooling stride)
int POOL_LAYERS[NUM_CONV_LAYERS][2] = { { 2, 2 } };
*/
// Conv layer config: (kernel size, number of kernels)
int CONV_LAYERS[NUM_CONV_LAYERS][2] = {};
// Pool layer config: (pooling size, pooling stride)
int POOL_LAYERS[NUM_CONV_LAYERS][2] = {};

// Grab matrix n out of the doubly flattened w
// (w is a flattened collection of matrices, each flattened)
float* grab_matrix(float* w, int n, int* n_rows, int* n_columns) {
    int ind = 0;
    int i;
grab_matrix_loop:    for (i = 0; i < n; i++) {
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
grab_matrix_dma_loop:    for (i = 1; i < layer; i++) {
        offset += layers[i].input_rows * layers[i].input_cols;
    }
    size_t size =
            layers[layer].input_rows * layers[layer].input_cols * sizeof(float);
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

void copy_zeropad(
        float* a, int a_height, int a_width, int pad, float* result) {
    int i, j;
    int result_width = a_width + 2*pad;
    int result_height = a_height + 2*pad;
    for (i = 0; i < pad; i++) {
        for (j = 0; j < result_width; j++) {
            result[sub2ind(i, j, result_width)] = 0;
        }
    }

    for (i = pad; i < a_height + pad; i++) {
        for (j = 0; j < pad; j++) {
            result[sub2ind(i, j, result_width)] = 0;
        }
        // Copy the original array.
        for (j = pad; j < a_width + pad; j++) {
            result[sub2ind(i, j, result_width)] = a[sub2ind(i - pad, j - pad, a_width)];
        }
        for (j = a_width + pad; j < result_width; j++) {
            result[sub2ind(i, j, result_width)] = 0;
        }
    }

    for (i = a_height + pad; i < result_height; i++) {
        for (j = 0; j < result_width; j++) {
            result[sub2ind(i, j, result_width)] = 0;
        }
    }
}

void convolution2d_no_padding(float* a,
                              float* kernels,
                              int a_height,
                              int a_width,
                              int k_idx,
                              int k_width,
                              float* result) {

    int i, j, k, l;
    int result_width = a_width - k_width + 1;
    int result_height = a_height - k_width + 1;

    // Convolution borders.
    int start_i = 0;
    int start_j = 0;
    int end_i = a_width - k_width + 1;
    int end_j = a_height - k_width + 1;

    // Which kernel to read and where to put the output feature map.
    int start_k = k_width*k_width*k_idx;
    int start_result = result_width * result_height * k_idx;

    float partial_sum;

    for (i = start_i; i < end_i; i++) {
        for (j = start_j; j < end_j; j++) {
            partial_sum = 0;
            for (k = 0; k < k_width; k++) {
                for (l = 0; l < k_width; l++) {
                    partial_sum +=
                            conv_float2fixed(a[sub2ind(i+k, j+l, a_width)]) *
                            conv_float2fixed(
                                    kernels[sub2ind(k, l, k_width) + start_k]);
                }
            }
            result[sub2ind(i, j, result_width) + start_result] = partial_sum;
        }
    }
}

void convolution2d_zeropad(float* a,
                           float* kernels,
                           int a_height,
                           int a_width,
                           int k_idx,
                           int k_width,
                           float* result,
                           float* result_temp) {
    int padding = k_width / 2;
    copy_zeropad(a, a_height, a_width, padding, result_temp);
    convolution2d_no_padding(result_temp, kernels, a_height + 2 * padding,
                             a_width + 2 * padding, k_idx, k_width, result);
}

// Multiply matrices a and b with given sizes and store into result_goes_here.
//
// We could do something tricky by switching the role of result and temp, to
// avoid copying but let's leave that for now.
//
// result_temp is used to ensure that weird things don't happen if
// result_goes_here overlaps with a or b.
void matrix_multiply(float* a,
                     float* b,
                     int a_height,
                     int a_width_b_height,
                     int b_width,
                     float* result_goes_here,
                     float* result_temp) {

    int i, j, k;
    float value;

    // Initialize to zero
    int size = a_height * b_width;
    clear_matrix(result_temp, size);

matmul0:    for (i = 0; i < a_height; i++) {
matmul1:        for (j = 0; j < b_width; j++) {
matmul2:            for (k = 0; k < a_width_b_height; k++) {
                value = conv_float2fixed(a[sub2ind(i, k, a_width_b_height)]) *
                        conv_float2fixed(b[sub2ind(k, j, b_width)]);
                result_temp[sub2ind(i, j, b_width)] =
                        conv_float2fixed(result_temp[sub2ind(i, j, b_width)] +
                                         conv_float2fixed(value));
            }
        }
    }
    copy_matrix(result_temp, result_goes_here, size);
}

// Multiply matrices a and b, assuming the last row of b are biases.
//
// Args:
//   a_height = height of A matrix.
//   b_height = height of the B matrix, which is also the width of the A matrix
//     + 1.
//   b_width = width of the B matrix.
void matrix_multiply_with_bias(float* a,
                               float* b,
                               int a_height,
                               int b_height,
                               int b_width,
                               float* result) {

    // a is hid, b is weights
    int i, j, k;
    float partial_sum;
    float value;

    int a_width = b_height - 1;

matmulb0: for (i = 0; i < a_height; i++) {
matmulb1: for (j = 0; j < b_width; j++) {
            // Initialize to zero
            partial_sum = 0;
matmulb2: for (k = 0; k < a_width; k++) {
                value = conv_float2fixed(a[sub2ind(i, k, a_width)]) *
                        conv_float2fixed(b[sub2ind(k, j, b_width)]);
                partial_sum += value;
            }
            // Add the bias (the index of the last row is the width of A).
            partial_sum += conv_float2fixed(b[sub2ind(a_width, j, b_width)]);
            result[sub2ind(i, j, b_width)] = partial_sum;
        }
    }
}

void matrix_multiply_with_bias_and_copy(float* a,
                                        float* b,
                                        int a_height,
                                        int b_height,
                                        int b_width,
                                        float* result_goes_here,
                                        float* result_temp) {
    int size = a_height * b_width;
    matrix_multiply_with_bias(
            a, b, a_height, b_height, b_width, result_temp);
    copy_matrix(result_temp, result_goes_here, size);
}

// Multiply the matrices a and b, but assume that b has been transposed.
//
// Args:
//   a_height = height of the A matrix.
//   b_height = height of the UNTRANSPOSED B matrix.
//   b_width = width of the UNTRANSPOSED B matrix.
void matrix_multiply_with_bias_transpose(float* a,
                                         float* b,
                                         int a_height,
                                         int b_height,
                                         int b_width,
                                         float* result) {

    // a is hid, b is weights
    int i, j, k;
    float partial_sum;
    float value;

    int a_width = b_height - 1;

matmulbt0: for (i = 0; i < a_height; i++) {
matmulbt1: for (j = 0; j < b_width; j++) {
            // Initialize to zero
            partial_sum = 0;
matmulbt2: for (k = 0; k < a_width; k++) {
                value = conv_float2fixed(a[sub2ind(i, k, a_width)]) *
                        conv_float2fixed(b[sub2ind(j, k, b_height)]);
                partial_sum += value;
            }
            // Add the bias.
            partial_sum += conv_float2fixed(b[sub2ind(j, a_width, b_height)]);
            result[sub2ind(i, j, b_width)] = partial_sum;
        }
    }
}

// Dispatch to the appropriate activation function.
void activation_fun(float* hid, int size, float* sigmoid_table) {
    if (ACTIVATION_FUN == 0) {
        RELU(hid, size);
    } else if (ACTIVATION_FUN == 1) {
        sigmoid_lookup(hid, size, sigmoid_table);
    } else {
        sigmoidn(hid, size);
    }
}

void conv_fwd(float* data,
              float* kernels,
              int* num_rows,
              int* num_columns,
              float* hid,
              float* hid_temp,
              float* sigmoid_table) {

    int k, l;

    int padding = 0;
    for (l = 0; l < NUM_CONV_LAYERS; l++) {
        // Zero pad every input feature map of this layer.
        // Without pooling layers, each input feature map would be the same
        // size as the input.
        copy_zeropad(data, num_rows[l], num_columns[l], padding, hid_temp);
        for (k = 0; k < NUM_KERNELS; k++) {
            convolution2d_no_padding(
                    hid_temp, kernels, INPUT_X, INPUT_Y, k, KERNEL_SIZE, hid);
        }
    }
    print_debug(hid, 28, 28, 28);
}

void run_layer(float* activations,
               float* weights,
               layer_t curr_layer,
               float* result,
               float* sigmoid_table,
               bool do_activation_func) {

    layer_type l_type = curr_layer.type;
    if (l_type == FC || l_type == OUTPUT) {
        MATRIX_MULTIPLY_WITH_BIAS(activations, weights, NUM_TEST_CASES,
                                  curr_layer.input_rows, curr_layer.input_cols,
                                  result);
    }

    PRINT_DEBUG(result, NUM_TEST_CASES, curr_layer.input_rows,
                curr_layer.input_cols);

    if (do_activation_func) {
        // Pass through activation function
        activation_fun(
                result, NUM_TEST_CASES * curr_layer.input_cols, sigmoid_table);

        PRINT_DEBUG(result, NUM_TEST_CASES, curr_layer.input_cols,
                    curr_layer.input_cols);
    }
}

// Does the forward predictive pass of a neural net.
// Returns a float array of class predictions in row major format of size
// num_test_cases*num_labels
void nnet_fwd(float* data,
              float* weights,
              layer_t* layers,
              int num_layers,
              float* hid,
              float* hid_temp,
              float* sigmoid_table) {

    int i, j, l;

    // Alternate between reading from/writing to hid and hid_temp so we can
    // avoid copying matrices.
    bool result_in_temp = false;
    bool do_activation_func = true;

    if (PRINT_DATA_AND_WEIGHTS) {
        printf("DATA:\n");
        for (i = 0; i < NUM_TEST_CASES; i++) {
            printf("Datum %d:\n", i);
            for (j = 0; j < INPUT_DIM; j++) {
                printf("%e, ", data[sub2ind(i, j, INPUT_DIM)]);
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
    dmaLoad(data, 0, 0, NUM_TEST_CASES * INPUT_DIM * sizeof(float));
    grab_matrix_dma(weights, l, layers);
#else
// Don't need to grab 0th matrix.
#endif

    ///////////////////////////////
    /////     FIRST LAYER      ////
    ///////////////////////////////

    // First layer must directly pass "data" as the function argument.
    run_layer(data, weights, layers[l], hid, sigmoid_table, do_activation_func);

    ////////////////////////////////
    ////    REMAINING LAYERS    ////
    ////////////////////////////////

    for (l = 2; l < num_layers; l++) {
        // Don't run the activation function on the last layer.
        do_activation_func = (l != num_layers - 1);

#ifdef DMA_MODE
        grab_matrix_dma(weights, l, layers);
#endif

        if (result_in_temp) {
          run_layer(hid_temp, weights, layers[l], hid, sigmoid_table, do_activation_func);
        } else {
          run_layer(hid, weights, layers[l], hid_temp, sigmoid_table, do_activation_func);
        }

        result_in_temp = !result_in_temp;
    }

#ifdef DMA_MODE
    if (result_in_temp)
        dmaStore(hid_temp, 0, 0, NUM_TEST_CASES * NUM_CLASSES * sizeof(float));
    else
        dmaStore(hid, 0, 0, NUM_TEST_CASES * NUM_CLASSES * sizeof(float));
#endif
}

size_t next_multiple(size_t request, size_t align) {
  size_t n = request/align;
  if (n == 0)
    return align;  // Return at least this many bytes.
  return (n+1)*align;
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
            next_multiple(
                    sizeof(layer_t) * (NUM_CONV_LAYERS * 2 + NUM_FC_LAYERS + 1),
                    CACHELINE_SIZE));
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

            if (LAYER_TYPES[i+1] == FC) {
                layers[i].output_rows = NUM_TEST_CASES;
                layers[i].output_cols = INPUT_X * INPUT_Y;
            } else {
                layers[i].output_rows = INPUT_Y;
                layers[i].output_cols = INPUT_X;
            }
        } else if (layers[i].type == CONV) {
            layers[i].c_kernel_size = CONV_LAYERS[i][0];
            layers[i].c_num_kernels = CONV_LAYERS[i][1];
            layers[i].input_rows = layers[i-1].output_rows;
            layers[i].input_cols = layers[i-1].output_cols;
            layers[i].output_rows = layers[i].input_rows;
            layers[i].output_cols = layers[i].input_cols;
            last_conv_layer++;
        } else if (layers[i].type == POOL_MAX) {
            // Assume that the first layer will not be a pooling layer.
            layers[i].p_size = POOL_LAYERS[last_pool_layer][0];
            layers[i].p_stride = POOL_LAYERS[last_pool_layer][1];
            layers[i].input_rows = layers[i-1].output_rows;
            layers[i].input_cols = layers[i-1].output_cols;
            layers[i].output_rows = ((layers[i].input_rows - layers[i].p_size) /
                                     layers[i].p_stride) + 1;
            layers[i].output_cols = ((layers[i].input_cols - layers[i].p_size) /
                                     layers[i].p_stride) + 1;
            last_pool_layer++;
        } else if (layers[i].type == FLATTEN) {
            layers[i].input_rows = layers[i-1].input_rows;
            layers[i].input_cols = layers[i-1].input_cols;
            // TODO: Is this right?
            layers[i].output_rows = layers[i].input_rows * layers[i].input_cols;
            layers[i].output_cols = 1;
        } else if (layers[i].type == FC) {
            layers[i].input_rows = layers[i-1].output_cols + 1;
            layers[i].input_cols = NUM_HIDDEN_UNITS[last_fc_layer];

            // The next layer's input rows is the number of this layer's columns + 1.
            layers[i].output_cols = layers[i].input_cols;
            layers[i].output_rows = -1;  // Not used.
            last_fc_layer++;
        } else if (layers[i].type == OUTPUT) {
            // Assume that the last layer must be fully connected.
            layers[i].input_rows = layers[i-1].output_cols + 1;
            layers[i].input_cols = NUM_CLASSES;

            layers[i].output_cols = NUM_CLASSES;
            layers[i].output_rows = -1;  // Not used.

            // This is the last layer. Break.
            total_layers = ++i;
            break;
        } else {
            printf("Layer %d is an unsupported type!\n", i);
            assert(false);
        }
    }

    // Helpfully print out the network topology.
    for (i = 0; i < total_layers; i++) {
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
            printf("  Num kernels: %d\n", layers[i].c_num_kernels);
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
        } else if (type == SOFTMAX) {
            printf("Softmax\n");
        } else if (type == FLATTEN) {
            printf("Flatten\n");
            printf("  Input size: %d x %d\n", layers[i].input_rows,
                   layers[i].input_cols);
            printf("  Output size: %d x %d\n", layers[i].output_rows,
                   layers[i].output_cols);
        } else if (type == INPUT) {
            printf("Input layer\n");
            printf("  Input size: %d x %d\n", layers[i].input_rows,
                   layers[i].input_cols);
            printf("  Output size: %d x %d\n", layers[i].output_rows,
                   layers[i].output_cols);
        } else if (type == OUTPUT) {
            printf("Output layer\n");
            printf("  Weights: %d x %d\n", layers[i].input_rows,
                   layers[i].input_cols);
        }
    }
    return total_layers;
}

// This is the thing that we want to be good at in hardware
int main(int argc, const char* argv[]) {
    // set random seed (need to #include <time.h>)
    srand(1);

    layer_t* layers;
    int total_layers = configure_network(&layers);

    int i;

    bool RANDOM_WEIGHTS = true;
    bool RANDOM_DATA = true;

    // Initialize weights, data, and labels.
    float* weights;
    int err;
    int w_size = get_total_num_weights(layers, total_layers);
    err = posix_memalign((void**)&weights, CACHELINE_SIZE,
                         next_multiple(w_size * sizeof(float), CACHELINE_SIZE));
    ASSERT_MEMALIGN(weights, err);
    init_weights(weights, layers, total_layers, RANDOM_WEIGHTS, TRANSPOSE_WEIGHTS);

    float* data;
    int* labels;
    float* kernels;
    size_t data_size = NUM_TEST_CASES * INPUT_DIM;
    size_t label_size = NUM_TEST_CASES;
    size_t kernel_size = NUM_KERNELS * KERNEL_SIZE * KERNEL_SIZE;
    err = posix_memalign(
            (void**)&data, CACHELINE_SIZE,
            next_multiple(data_size * sizeof(float), CACHELINE_SIZE));
    ASSERT_MEMALIGN(data, err);
    err = posix_memalign(
            (void**)&labels, CACHELINE_SIZE,
            next_multiple(label_size * sizeof(int), CACHELINE_SIZE));
    ASSERT_MEMALIGN(labels, err);
    err = posix_memalign(
            (void**)&kernels, CACHELINE_SIZE,
            next_multiple(kernel_size * sizeof(float), CACHELINE_SIZE));
    ASSERT_MEMALIGN(kernels, err);
    init_data(data, NUM_TEST_CASES, INPUT_DIM, RANDOM_DATA);
    init_labels(labels, NUM_TEST_CASES, RANDOM_DATA);
    init_kernels(kernels, kernel_size);
    printf("Data has %lu elements.\n", data_size);

    // Get the dimensions of the biggest matrix that will ever come out of
    // matrix_multiply. All of them will have NUM_TEST_CASES rows. So I just
    // find the biggest number of columns.
    printf("Setting up arrays\n");
    int biggest_cols = 0;
    for (i = 1; i < total_layers; i++) {
        if (layers[i].input_cols > biggest_cols)
            biggest_cols = layers[i].input_cols;
    }
    printf("Largest hidden/output layer: %d\n", biggest_cols);
    fflush(stdout);

    // Then, allocate memory for it. We will always place the result of our
    // matrix multiplications in here.
    //
    // Mapped to its own scratchpad.
    float* hid;
    float* hid_temp;
    size_t hid_size = NUM_TEST_CASES * biggest_cols;
    hid_size = NUM_TEST_CASES * 34 * 34;
    err = posix_memalign(
            (void**)&hid, CACHELINE_SIZE,
            next_multiple(hid_size * sizeof(float), CACHELINE_SIZE));
    ASSERT_MEMALIGN(hid, err);
    err = posix_memalign(
            (void**)&hid_temp, CACHELINE_SIZE,
            next_multiple(hid_size * sizeof(float), CACHELINE_SIZE));
    ASSERT_MEMALIGN(hid_temp, err);

    // This file is not looked at by aladdin so malloc is fine.
    // If I do the old version then I get a memory overflow, because the
    // max stack size is not big enough for TIMIT stuff.

    // Build the sigmoid lookup table
    // May want to change this to be "non-centered"
    // to avoid (sigmoid_coarseness - 1.0)
    // so we can use bit shift in lookup function with fixed point precisions
    printf("Setting up sigmoid lookup table...\n");
    int sigmoid_coarseness = 1 << LG_SIGMOID_COARSENESS;
    float sigmoid_table[sigmoid_coarseness];
    float sig_step = (float)(SIG_MAX - SIG_MIN) / (sigmoid_coarseness - 1.0);
    float x_sig = (float)SIG_MIN;
    for (i = 0; i < sigmoid_coarseness; i++) {
        sigmoid_table[i] = conv_float2fixed(1.0 / (1.0 + exp(-x_sig)));
        // printf("%f, %f\n", x_sig, sigmoid_table[i]);
        x_sig += sig_step;
    }

    // -------------------------------------------------------- //
    //     THIS IS THE FUNCTION BEING SIMULATED IN HARDWARE     //
    // -------------------------------------------------------- //
#ifdef GEM5_HARNESS
    mapArrayToAccelerator(
            INTEGRATION_TEST, "data", data, data_size * sizeof(float));
    mapArrayToAccelerator(
            INTEGRATION_TEST, "weights", weights, w_size * sizeof(float));
    mapArrayToAccelerator(
            INTEGRATION_TEST, "hid", hid, hid_size * sizeof(float));
    mapArrayToAccelerator(
            INTEGRATION_TEST, "hid_temp", hid_temp, hid_size * sizeof(float));
    // sigmoid_table, num_units, num_fc_rows, and num_columns I consider as
    // configuration, which is really one-time (compared to data and weights
    // which may need to be reloaded multiple times for the same network).
    // They're small enough that they should be completely partitioned, and
    // because they're configuration time parameters, I don't count them as
    // DMA.
    invokeAcceleratorAndBlock(INTEGRATION_TEST);
#else
    // Run a forward pass through the neural net
    printf("Running forward pass\n");
    /*
    conv_fwd(data, kernels, num_conv_rows, num_conv_columns, hid,
             hid_temp, sigmoid_table);
             */
    // The function being synthesized
    nnet_fwd(data, weights, layers, total_layers, hid, hid_temp, sigmoid_table);
#endif

    // "hid" now contains the outputs

    // Print the result, maybe not all the test_cases
    int num_to_print = 1;
    // don't try to print more test cases than there are
    num_to_print =
            num_to_print < NUM_TEST_CASES ? num_to_print : NUM_TEST_CASES;

    // Compute the classification error rate
    float* result = NUM_FC_LAYERS % 2 == 0 ? hid : hid_temp;
    int num_errors = 0;
    for (i = 0; i < NUM_TEST_CASES; i++) {
        if (arg_max(result + i * NUM_CLASSES, NUM_CLASSES, 1) != labels[i]) {
            num_errors = num_errors + 1;
        }
    }
    FILE* output_labels = fopen("output_labels.out", "w");
    for (i = 0; i < NUM_TEST_CASES; i++) {
        fprintf(output_labels, "Test %d label: %d\n", i,
                arg_max(result + i * NUM_CLASSES, NUM_CLASSES, 1));
    }
    fclose(output_labels);
    float error_fraction = ((float)num_errors) / ((float)NUM_TEST_CASES);
    printf("Fraction incorrect (over %d cases) = %f\n", NUM_TEST_CASES,
           error_fraction);

    // Write this number to a file
    FILE* accuracy_file;
    accuracy_file = fopen("accuracy.txt", "w");
    fprintf(accuracy_file, "%f", error_fraction);
    fclose(accuracy_file);

    free(hid);
    free(hid_temp);
    free(weights);
    free(data);
    free(labels);
    free(kernels);
    free(layers);
}
