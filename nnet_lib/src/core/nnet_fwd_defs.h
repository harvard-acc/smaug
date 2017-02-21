#ifndef _NNET_FWD_DEFS_H_
#define _NNET_FWD_DEFS_H_

// This is the core header of nnet_lib.

extern int NUM_TEST_CASES;
extern int NUM_CLASSES;
extern int INPUT_DIM;

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
    // Fully connected layer.
    FC,
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

  // For CONV and POOL layers
  int field_size;
  int field_stride;

  // CONV layers only.
  int c_padding;

  // Where are the class predictions stored, hid or hid_temp?
  int result_in_temp;
} layer_t;

// Execute curr_layer on the provided activations and weights.
bool run_layer(float* activations,
               float* weights,
               layer_t curr_layer,
               float* result_temp,
               float* sigmoid_table,
               bool do_activation_func);

// Possible values of ARCHITECTURE.
//
// This defines the structure of the nnet accelerator - whether it is a
// monolithic block or a collection of multiple blocks.
//
// Allowed values are: MONOLITHIC, COMPOSABLE.
#define MONOLITHIC 0
#define COMPOSABLE 1

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

#endif
