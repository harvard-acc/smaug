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
#define INPUT_Z 1
#define INPUT_DIM (INPUT_X * INPUT_Y)
#define NUM_CLASSES 10
// number of stored points in sigmoid lookup table
#define LG_SIGMOID_COARSENESS 4
#define NUM_TEST_CASES 2      // NOT READ BY nnet_fwd.c, ONLY BY the other one
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
#ifndef TREE_MAX
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
// Operation: data[row][col]
#define sub2ind(r, c, n_columns) ((r) * (n_columns) + (c))

// 3D indexing into a flattened array.
//
// Operation: data[height][row][col]
#define sub3ind(h, r, c, n_rows, n_cols)                                       \
    (sub2ind(r, c, n_cols) + (h) * ((n_rows) * (n_cols)))

// 4D indexing into a flattened array.
//
// Operation: data[depth][height][row][col]
//
//                   c
//              ------------
//           r /           /|
//            /           / |
//           /           /  |
//  _     _  ------------   |
//  |     |  |          |   /
//  |     h  |          |  /|
//  |     |  |          | / |
//  d     -  ------------/  |
//  |        |          |   /
//  |        |          |  /
//  |        |          | /
//  -        |-----------/
//
// n_hgt = maximum value of h
// n_rows = maximum value of r
// n_cols = maximum value of c
//
// As an example, this is used to index input feature maps in convolutional
// layers, where depth = number of input images, and height = number of feature
// maps from previous layer.
#define sub4ind(d, h, r, c, n_hgt, n_rows, n_cols)                             \
    (sub3ind(h, r, c, n_rows, n_cols) + (d) * ((n_rows) * (n_cols) * (n_hgt)))

#define printf_sub4ind(d, h, r, c, n_hgt, n_rows, n_cols)                      \
    printf("sub4ind(%d, %d, %d, %d, %d, %d, %d) = %d\n", d, h, r, c, n_hgt,    \
           n_rows, n_cols, (sub3ind(h, r, c, n_rows, n_cols) +                 \
                            (d) * ((n_rows) * (n_cols) * (n_hgt))))

#if DEBUG == 1
#define PRINT_DEBUG(hid, rows, cols, num_cols)                                 \
    print_debug(hid, rows, cols, num_cols)
#define PRINT_DEBUG4D(hid, rows, cols, height)                                 \
    print_debug4d(hid, rows, cols, height)
#define PRINT_MSG(msg) printf(msg);
#else
#define PRINT_DEBUG(hid, rows, cols, num_cols)
#define PRINT_DEBUG4D(hid, rows, cols, height)
#define PRINT_MSG(msg)
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

typedef enum _data_init_mode {
    RANDOM,    // Generate pseudo-random input.
    FIXED,     // Use (mostly) constant values (helpful for debugging).
    READ_FILE  // Read data and weights from files.
} data_init_mode;

typedef enum _layer_type {
    // 2D convolutional layer.
    CONV,
    // Max pooling layer.
    POOL_MAX,
    // Average pooling layer.
    POOL_AVG,
    // Softmax output.
    SOFTMAX,
    // Flatten previous output into a column vector (for FC layers).
    FLATTEN,
    // Fully connected layer.
    FC,
    // First layer (skipped during execution).
    INPUT,
    // Output label layer, fully connected (the implicit last layer).
    OUTPUT,
    // End the network without a FC output layer.  This is mostly used for
    // debugging.
    END
} layer_type;

// Description of a layer in a neural network.
//
// TODO: Due to Aladdin's requirement to specify a word size for arrays, all
// members of a struct must be of the same size (so that they can be
// partitioned at a single granularity). This needs to be fixed so that we can have
// struct members of different sizes.
typedef struct _layer_t {
  // Type of layer.
  layer_type type;

  // Data input/output dimensions on a per iteration basis.
  //
  // These values refer to a single data point or image, so the total size of
  // the layer's output is output_rows * output_cols * NUM_TEST_CASES.  Our
  // convention is that layer i gets its input row/col from layer i-1's output
  // row/col. Depth is the number of feature maps read/written per iteration.
  //
  // Input/output rows/cols:
  //
  //   Conv/pool layers: the dimensions of the input/output images/activations.
  //      Note that the activations are stored in row vector form.
  //   FC layers: input_rows/cols is the size of the weights matrix. Output
  //      cols is the number of input rows for the next layer. Output rows is 1.
  //   Input layer: input rows/cols are the dimensions of each input image.
  //      Output rows/cols are the dimensions of the transformed input to the
  //      next layer.
  //
  // Input/output height:
  //    Conv layers: input height is the number of input feature maps from the
  //      previous layer, and output height is the number of filters (aka number
  //      of output feature maps).
  //    Pool layers: input/output heights are equal to number of input feature maps.
  //    All other layers: 1.
  int input_rows;
  int input_cols;
  int input_height;
  int output_rows;
  int output_cols;
  int output_height;

  // for CONV layers only.
  int c_kernel_size;

  // for POOL layers only.
  int p_size;
  int p_stride;

  // Where are the class predictions stored, hid or hid_temp?
  int result_in_temp;
} layer_t;

#endif
