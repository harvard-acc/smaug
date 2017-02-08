#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <float.h>

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

// Network layer configuration.
layer_type LAYER_TYPES[MAX_LAYERS] = {
    INPUT, CONV, POOL_MAX, FLATTEN, FC, FC, FC, OUTPUT
};
// Fully connected layer config.
int NUM_HIDDEN_UNITS[NUM_FC_LAYERS] = { 256, 256, 256 };

// Conv layer config: (kernel size, number of kernels)
int CONV_LAYERS[NUM_CONV_LAYERS][2] = { { 3, 1 } };
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

// Zeropad each image in @a by @pad zeros.
//
// a is a matrix of flattened image vectors with dimensions NUM_TEST_CASES * n
// * m, where n and m are denoted by curr_layer.output_rows and
// curr_layer.output_cols. Yes, "output_rows" and "output_cols". "input_rows"
// and "input_cols" is the dimensions of the data after zeropadding because
// this is considered as the "input" to the convolution itself.
//
// Place the result in @result, which is an array that is assumed to be large
// enough for this operation.
void copy_zeropad(float* a, layer_t curr_layer, int pad, float* result) {
    int i, j, ni;

    int a_height = curr_layer.output_rows;
    int a_width = curr_layer.output_cols;
    int result_width = a_width + 2 * pad;
    int result_height = a_height + 2 * pad;

copy_zeropad_outer:
    for (ni = 0; ni < NUM_TEST_CASES; ni++) {
    copy_zeropad_first:
        for (i = 0; i < pad; i++) {
        copy_zeropad_first_cols:
            for (j = 0; j < result_width; j++) {
                result[sub3ind(i, j, ni, result_height, result_width)] = 0;
            }
        }

    copy_zeropad_left:
        for (i = pad; i < a_height + pad; i++) {
        copy_zeropad_left_cols:
            for (j = 0; j < pad; j++) {
                result[sub3ind(i, j, ni, result_height, result_width)] = 0;
            }
        // Copy the original array.
        copy_zeropad_copy_cols:
            for (j = pad; j < a_width + pad; j++) {
                result[sub3ind(i, j, ni, result_height, result_width)] =
                        a[sub3ind(i - pad, j - pad, ni, a_height, a_width)];
            }
        copy_zeropad_right_cols:
            for (j = a_width + pad; j < result_width; j++) {
                result[sub3ind(i, j, ni, result_height, result_width)] = 0;
            }
        }

    copy_zeropad_last:
        for (i = a_height + pad; i < result_height; i++) {
        copy_zeropad_last_cols:
            for (j = 0; j < result_width; j++) {
                result[sub3ind(i, j, ni, result_height, result_width)] = 0;
            }
        }
    }
}

// Downsample the input using a max-pooling operation.
//
// @input contains a stack of 2D images.
// The parameters of the pooling operation are given in @curr_layer.
//
// The downsampled result is placed into @result.
void max_pooling(float* input, float* result, layer_t curr_layer) {
    int i, j, k, l, ni, oi, oj;
    float in_val, curr_max;
    int stride = curr_layer.p_stride;
    int size = curr_layer.p_size;
    int height = curr_layer.input_rows;
    int width = curr_layer.input_cols;

#if TREE_MAX == 1
    int total_pool_size = size * size;
    float elems[total_pool_size];
    int elem_idx;
#endif

maxpool_outer:
    for (ni = 0; ni < NUM_TEST_CASES; ni++) {
        // Output image indices.
        oi = 0;
        oj = 0;
    maxpool_input_rows:
        for (i = 0; i < curr_layer.input_rows; i += stride) {
        maxpool_input_cols:
            for (j = 0; j < curr_layer.input_cols; j += stride) {
                // Iterate over the pooling field.
#if TREE_MAX == 1
                elem_idx = 0;
            maxpool_tree_outer:
                for (k = 0; k < size; k++) {
                maxpool_tree_inner:
                    for (l = 0; l < size; l++) {
                         elems[elem_idx] = input[sub3ind(i+k, j+l, ni, height, width)];
                         elem_idx++;
                    }
                }

                if (total_pool_size == 4)
                    curr_max = max4(elems[0], elems[1], elems[2], elems[3]);
                else if (total_pool_size == 9)
                    curr_max = max9(elems[0], elems[1], elems[2], elems[3],
                                    elems[4], elems[5], elems[6], elems[7],
                                    elems[8]);
                else
                    assert(false && "Unsupported pooling size!");

#else
                curr_max = -FLT_MAX;
            maxpool_iter_outer:
                for (k = 0; k < size; k++) {
                maxpool_iter_inner:
                    for (l = 0; l < size; l++) {
                        in_val = input[sub3ind(i+k, j+l, ni, height, width)];
                        curr_max = max(in_val, curr_max);
                    }
                }
#endif

                result[sub3ind(oi, oj, ni, curr_layer.output_rows,
                               curr_layer.output_cols)] = curr_max;
                oj++;
            }
            oi ++;
            oj = 0;
        }
    }
}

// Perform a 2D convolution operation over the data in @a.
//
// The convolutions are specified through the curr_layers struct. Zero padding
// for borders is not handled.
//
// @a contains a stack of 2D images, and kernels contains a stack of 2D
// kernels, whose dimensions are specified in @curr_layer.
//
// The result of the operation is placed into @result, which is assumed to be
// large enough to hold the data. @result is also a stack of 2D images.
void convolution2d_no_padding(float* a,
                              float* kernels,
                              layer_t curr_layer,
                              float* result) {

    int i, j, k, l;
    int ni, nk;
    float a_val, kern_val;

    int a_height = curr_layer.input_rows;
    int a_width = curr_layer.input_cols;
    int k_width = curr_layer.c_kernel_size;

    int result_height = curr_layer.output_rows;
    int result_width = curr_layer.output_cols;

    // Convolution borders.
    int start_i = 0;
    int start_j = 0;
    int end_i = result_width;
    int end_j = result_height;

    float partial_sum;

// Loop over all input activation feature maps.
conv2d_outer:
    for (ni = 0; ni < NUM_TEST_CASES; ni++) {
    // Convolution loop over the output pixels.
    conv2d_input_rows:
        for (i = start_i; i < end_i; i++) {
        conv2d_input_cols:
            for (j = start_j; j < end_j; j++) {

            // For each kernel in this layer.
            conv2d_kernel_outer:
                for (nk = 0; nk < curr_layer.c_num_kernels; nk++) {

                    // Convolution loop over the kernel.
                    partial_sum = 0;
                conv2d_kernel_rows:
                    for (k = 0; k < k_width; k++) {
                    conv2d_kernel_cols:
                        for (l = 0; l < k_width; l++) {
                            a_val = conv_float2fixed(a[sub3ind(
                                    i + k, j + l, ni, a_height, a_width)]);
                            kern_val = conv_float2fixed(kernels[sub3ind(
                                    k, l, nk, k_width, k_width)]);
                            partial_sum += conv_float2fixed(a_val * kern_val);
                        }
                    }
                    result[sub3ind(i, j, ni, result_height, result_width)] =
                            partial_sum;
                }
            }
        }
    }
}

// Perform a 2D convolution on the data in @input with zero padding.
//
// The amount of padding is determined by the convolution operation configured
// in the @curr_layer struct.
//
// NOTE: The data is not actually placed in @result, but rather into @input!
// This is because the input data is zero-padded and copied into result. To
// avoid a second copy back into @input, we simply execute the convolution using
// @result as the input array instead.
void convolution2d_zeropad(float* input,
                           float* kernels,
                           layer_t curr_layer,
                           float* result) {
    int padding = (curr_layer.c_kernel_size - 1) / 2;
    copy_zeropad(input, curr_layer, padding, result);
    convolution2d_no_padding(result, kernels, curr_layer, input);
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

matmul0:
    for (i = 0; i < a_height; i++) {
    matmul1:
        for (j = 0; j < b_width; j++) {
        matmul2:
            for (k = 0; k < a_width_b_height; k++) {
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

matmulb0:
    for (i = 0; i < a_height; i++) {
    matmulb1:
        for (j = 0; j < b_width; j++) {
            // Initialize to zero
            partial_sum = 0;
        matmulb2:
            for (k = 0; k < a_width; k++) {
                value = conv_float2fixed(a[sub2ind(i, k, a_width)]) *
                        conv_float2fixed(b[sub2ind(k, j, b_width)]);
                partial_sum += value;
                // printf("partial_sum: %f\n", partial_sum);
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

matmulbt0:
    for (i = 0; i < a_height; i++) {
    matmulbt1:
        for (j = 0; j < b_width; j++) {
            // Initialize to zero
            partial_sum = 0;
        matmulbt2:
            for (k = 0; k < a_width; k++) {
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

bool run_layer(float* activations,
               float* weights,
               layer_t curr_layer,
               float* result_temp,
               float* sigmoid_table,
               bool do_activation_func) {
    bool result_in_input = false;
    layer_type l_type = curr_layer.type;
    if (l_type == FC) {
        MATRIX_MULTIPLY_WITH_BIAS(activations, weights, NUM_TEST_CASES,
                                  curr_layer.input_rows, curr_layer.input_cols,
                                  result_temp);
        PRINT_DEBUG(
                result_temp, 1, curr_layer.output_cols, curr_layer.output_cols);
    } else if (l_type == CONV) {
        convolution2d_zeropad(activations, weights, curr_layer, result_temp);
        PRINT_DEBUG(activations, curr_layer.output_rows, curr_layer.output_cols,
                    curr_layer.output_cols);
        result_in_input = true;
    } else if (l_type == POOL_MAX) {
        max_pooling(activations, result_temp, curr_layer);
        PRINT_DEBUG(result_temp, curr_layer.output_rows, curr_layer.output_cols,
                    curr_layer.output_cols);
    } else if (l_type == FLATTEN) {
        // This is just a dummy layer. Return.
        result_in_input = true;
        do_activation_func = false;
    }

    if (do_activation_func) {
        // Pass through activation function
        if (result_in_input) {
            activation_fun(activations,
                           curr_layer.output_rows * curr_layer.output_cols,
                           sigmoid_table);

            PRINT_DEBUG(activations, curr_layer.output_rows, curr_layer.output_cols,
                        curr_layer.output_cols);
        } else {
            activation_fun(result_temp,
                           curr_layer.output_rows * curr_layer.output_cols,
                           sigmoid_table);

            PRINT_DEBUG(result_temp, curr_layer.output_rows,
                        curr_layer.output_cols, curr_layer.output_cols);
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

    layers[num_layers - 1].result_in_temp = result_in_temp;
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
            usage = layer.input_rows * layer.input_cols;
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

            if (LAYER_TYPES[i+1] == FC) {
                layers[i].output_rows = 1;
                layers[i].output_cols = INPUT_X * INPUT_Y;
            } else {
                layers[i].output_rows = INPUT_Y;
                layers[i].output_cols = INPUT_X;
            }
        } else if (layers[i].type == CONV) {
            layers[i].c_kernel_size = CONV_LAYERS[last_conv_layer][0];
            layers[i].c_num_kernels = CONV_LAYERS[last_conv_layer][1];
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
            layers[i].output_rows = ((layers[i].input_rows - layers[i].p_size) /
                                     layers[i].p_stride) + 1;
            layers[i].output_cols = ((layers[i].input_cols - layers[i].p_size) /
                                     layers[i].p_stride) + 1;
            last_pool_layer++;
        } else if (layers[i].type == FLATTEN) {
            layers[i].input_rows = layers[i-1].output_rows;
            layers[i].input_cols = layers[i-1].output_cols;
            // TODO: Is this right?
            layers[i].output_rows = 1;
            layers[i].output_cols = layers[i].input_rows * layers[i].input_cols;
        } else if (layers[i].type == FC) {
            layers[i].input_rows = layers[i-1].output_cols + 1;
            layers[i].input_cols = NUM_HIDDEN_UNITS[last_fc_layer];

            // The next layer's input rows is the number of this layer's columns + 1.
            layers[i].output_cols = layers[i].input_cols;
            layers[i].output_rows = 1;
            last_fc_layer++;
        } else if (layers[i].type == OUTPUT) {
            // Assume that the last layer must be fully connected.
            layers[i].type = FC;
            layers[i].input_rows = layers[i-1].output_cols + 1;
            layers[i].input_cols = NUM_CLASSES;

            layers[i].output_cols = NUM_CLASSES;
            layers[i].output_rows = 1;

            // This is the last layer. Break.
            total_layers = ++i;
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
        }
    }
    return total_layers;
}

// This is the thing that we want to be good at in hardware
int main(int argc, const char* argv[]) {
    int i, err;

    // set random seed (need to #include <time.h>)
    srand(1);

    layer_t* layers;
    int total_layers = configure_network(&layers);
    printf("Size of layer configuration: %lu\n", total_layers * sizeof(layer_t));

    bool RANDOM_WEIGHTS = true;
    bool RANDOM_DATA = true;

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
    free(labels);
    free(layers);
}
