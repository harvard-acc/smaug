#ifndef _NNET_FWD_H_
#define _NNET_FWD_H_

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Fixed parameters
#define INPUT_X 28
#define INPUT_Y 28
#define INPUT_DIM (INPUT_X * INPUT_Y)
#define NUM_CLASSES 10
// number of stored points in sigmoid lookup table
#define LG_SIGMOID_COARSENESS 4
#define NUM_TEST_CASES 1      // NOT READ BY nnet_fwd.c, ONLY BY the other one
#define SIG_MIN -5            // lower input bound for sigmoid lookup table
#define SIG_MAX +5            // upper input bound for sigmoid lookup table

// Parameters for optimization
#define NUM_FC_LAYERS 3
#define NUM_CONV_LAYERS 1
#define MAX_LAYERS ((NUM_CONV_LAYERS)*2 + NUM_FC_LAYERS + 3)

#define ACTIVATION_FUN 0  // categorical, 0=RELU 1=sigmoid lookup 2=true sigmoid
#define NUM_OF_INT_BITS                                                        \
    6  // number of bits before the decimal pt in our representation
#define NUM_OF_FRAC_BITS                                                       \
    26  // number of bits after the decimal pt in our representation

// If 1, then this transposes the data in the weights matrix such that the
// access pattern is strided in the same way as the activations, which is
// beneficial for increasing memory level parallelism (by reducing the number
// of consecutive references to the same partition of a scratchpad).
//
// This can also be defined from the build command.
#ifndef TRANSPOSE_WEIGHTS
#define TRANSPOSE_WEIGHTS 0
#endif

// If 1, then this uses a tree-based max implementation for the pooling layers,
// which is more efficient than a loop.
#ifdef TREE_MAX
#define TREE_MAX 1
#endif

// Turns out debugging output, which prints out the results of operations.
#ifndef DEBUG
#define DEBUG 0
#endif

// Print the input data and the complete set of weights.
#ifndef PRINT_DATA_AND_WEIGHTS
#define PRINT_DATA_AND_WEIGHTS 0
#endif

//////////////////////////////////////////////
////   USER TUNABLE PARAMETERS END HERE   ////
//////////////////////////////////////////////

// Macros for computing the maximum of a group of elements.
//
// Why macros and not functions (or a loop)? A loop takes O(n) cycles to
// compute the maximum, when it could be done in O(log n) time with a tree
// based implementation. But Aladdin regards function calls as a hard
// dependency that it does not optimize across, so we would not get the
// parallelism we expect from the tree. Thus, these must be macros.
//
// I've only implemented a few of these. These are only meant for the pooling
// layers, and we shouldn't need more than a 3x3 pooling layer anyways.
#define max(A, B) (((A) > (B)) ? (A) : (B))
#define max4(e0, e1, e2, e3) max(max(e0, e1), max(e2, e3))
#define max8(e0, e1, e2, e3, e4, e5, e6, e7)                                   \
    max(max4(e0, e1, e2, e3), max4(e4, e5, e6, e7))
#define max9(e0, e1, e2, e3, e4, e5, e6, e7, e8)                               \
    max(max8(e0, e1, e2, e3, e4, e5, e6, e7), e8)

// Based on whether the weights matrix is transposed or not, use a different
// multiplication kernel.
#if TRANSPOSE_WEIGHTS == 1
#define MATRIX_MULTIPLY_WITH_BIAS matrix_multiply_with_bias_transpose
#else
#define MATRIX_MULTIPLY_WITH_BIAS matrix_multiply_with_bias
#endif

// 2D indexing into a flattened array.
//
// This assumes data is stored in row major order.
#define sub2ind(r, c, n_columns) ((r) * (n_columns) + (c))

// 3D indexing into a flattened array.
//
// This assumes the data structure is a stack of 2D matrices, where h is the
// desired depth of the stack.
#define sub3ind(r, c, h, n_rows, n_cols)                                       \
    (sub2ind(r, c, n_cols) + h * (n_rows * n_cols))

#if DEBUG == 1
#define PRINT_DEBUG(hid, rows, cols, num_cols)                                 \
    print_debug(hid, rows, cols, num_cols)
#else
#define PRINT_DEBUG(hid, rows, cols, num_cols)
#endif

#define CACHELINE_SIZE 32

#define ASSERT_MEMALIGN(ptr, err) \
    assert(err == 0 && "Failed to allocate memory for " #ptr ".\n");

#define INPUTS_FILENAME                                                        \
    "/home/jmh/projects/pesc_hardware/HardwareNets/../mnist/"                  \
    "validation_data_textual_all_10000.txt"
#define LABELS_FILENAME                                                        \
    "/home/jmh/projects/pesc_hardware/HardwareNets/../mnist/"                  \
    "validation_labels_textual_all_10000_not_one_hot.txt"
#define WEIGHTS_FILENAME                                                       \
    "/home/jmh/projects/pesc_hardware/HardwareNets/../mnist/"                  \
    "mnist_textual_weights.txt"

typedef struct _conv2d_layer {
  int layer_idx;
  int num_kernels;
  int kernel_size;
  int input_width;
  int input_height;
} conv2d_layer;

typedef struct _fc_layer {
  int layer_idx;
  int num_rows;
  int num_columns;
} fc_layer;

typedef enum _layer_type {
    CONV,
    POOL_MAX,
    POOL_AVG,
    // TODO: Remove.
    FLATTEN,  // Flatten previous output into a column vector (for FC layers).
    FC,
    INPUT,  // First layer (skipped during execution).
    OUTPUT,  // Output label layer (and also the implicit last layer).
    SOFTMAX
} layer_type;

typedef struct _layer_t {
  // Type of layer.
  layer_type type;

  // Data input/output dimensions on a per iteration basis.
  //
  // These values refer to a single data point or image, so the total size of
  // the layer's output is output_rows * output_cols * NUM_TEST_CASES.  Our
  // convention is that layer i gets its input row/col from layer i-1's output
  // row/col.
  //
  // Conv/pool layers: the dimensions of the input/output images/activations.
  //    Note that the activations are stored in row vector form.
  // FC layers: input_rows/cols is the size of the weights matrix. Output cols
  //    is the number of input rows for the next layer. Output rows is 1.
  // Input layer: input rows/cols are the dimensions of each input image.
  //    Output rows/cols are the dimensions of the transformed input to the
  //    next layer.
  int input_rows;
  int input_cols;
  int output_rows;
  int output_cols;

  // for CONV layers only.
  int c_num_kernels;
  int c_kernel_size;

  // for POOL layers only.
  int p_size;
  int p_stride;

  // Where are the class predictions stored, hid or hid_temp?
  bool result_in_temp;
} layer_t;

#endif
