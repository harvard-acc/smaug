#ifndef _NNET_FWD_DEFS_H_
#define _NNET_FWD_DEFS_H_

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

// This is the core header of nnet_lib.

typedef enum _data_init_mode {
    RANDOM,      // Generate pseudo-random input.
    FIXED,       // Use (mostly) constant values (helpful for debugging).
    FAST_FIXED,  // Use completely constant values (fast simulation).
    READ_FILE    // Read data and weights from files.
} data_init_mode;

typedef enum _pool_type {
    MAX,
    AVG,
    NUM_POOLING_TYPES,
} pool_type;

typedef enum _activation_type {
    NO_ACTIVATION,
    RELU,
    RELU_THRESHOLD,
    LRELU,
    ELU,
    SELU,
    TANH,
    HARD_TANH,
    SIGMOID,
    SOFTMAX
} activation_type;

typedef enum _input_pp {
    FLATTEN,
    UNFLATTEN,
    NCHW_TO_NHWC,
    NO_PREPROCESSING,
} input_pp;

typedef enum _layer_type {
    // Standard 3D convolutional layer.
    CONV_STANDARD,
    // Depthwise convolution layer.
    CONV_DEPTHWISE,
    // Pointwise convolution layer.
    CONV_POINTWISE,
    // Pooling layer.
    POOLING,
    // Fully connected layer.
    FC,
    // Batch normalization layer.
    BATCH_NORM,
    // Output label layer, fully connected (the implicit last layer).
    OUTPUT,
    // Input layer. No actual work is done on this layer.
    INPUT,
    // End the network without a FC output layer.  This is mostly used for
    // debugging.
    END
} layer_type;

typedef struct _dims_t {
    int rows;
    int cols;
    int height;
    int align_pad;
} dims_t;

typedef enum _io_req_t {
    IO_NONE = 0,
    IO_DMA = 1,
    IO_ACP = 2,
    IO_CACHE = 3,
} io_req_t;

typedef enum _bn_weights_idx {
    MeanIndex,
    VarianceIndex,
    GammaIndex,
    ScaleshiftIndex = GammaIndex,  // for MKL.
    BetaIndex,
    NumWeightTypes
} bn_weights_idx;

typedef enum _sigmoid_impl_t {
    ExpUnit,
    CenteredLUT,
    NoncenteredLUT,
    NumSigmoidImpls
} sigmoid_impl_t;

// Wraps a dynamically allocated array (d for data) and its size (number of
// elements, not bytes).
// TODO: Use int for all these sizes, not size_t.
typedef struct _farray_t {
    float* d;
    size_t size;
    // If true, this buffer contains a dynamically-allocated pointer and can be
    // freed. If not, you should NOT try to call free() on this buffer.
    bool freeable;
} farray_t;

typedef struct _iarray_t {
    int* d;
    size_t size;
} iarray_t;

// We store packed half-precision floating-point in 32-bit chunks.
typedef uint32_t packed_fp16;
typedef uint16_t float16;
typedef struct _fp16array_t {
    packed_fp16* d;
    size_t size;
    bool freeable;
} fp16array_t;

typedef enum _data_storage_t {
    Uncompressed = 0,
    CSR = 1,
    PackedCSR = 2,
    UncompressedHalfPrecision = 3,
    NumDataStorageTypes,
} data_storage_t;

// Valid storage formats for any kind of data (weights or activations):
union DataFormat {
    struct _packed_csr_array_t* packed;
    struct _csr_array_t* csr;
    farray_t* dense;
    fp16array_t* dense_hp;
};

typedef struct _data_list {
    // A pointer to dynamically allocated storage, holding pointers to
    // pointers to weights.
    union DataFormat* data;
    // Indicates the weights storage format.
    data_storage_t* type;
    int len;
} data_list;

typedef struct _padding {
    int top;
    int bottom;
    int right;
    int left;
} padding;

typedef struct _stride_dims {
    int rows;
    int cols;
} stride_dims;

typedef enum _data_mvmt_policy {
    DmaAlways,
    AcpAlways,
    AcpIfWeightsAreReused,
    NumDataMvmtPolicies
} data_mvmt_policy;

// When ping-ponging data between two buffers, use this to indicate which one
// stores the last output (and the next input).
typedef data_list* result_buf;

// Description of a layer in a neural network.
//
// TODO: Due to Aladdin's requirement to specify a word size for arrays, all
// members of a struct must be of the same size (so that they can be
// partitioned at a single granularity). This needs to be fixed so that we can have
// struct members of different sizes.
typedef struct _layer_t {
  // Type of layer.
  layer_type type;
  // Layer number.
  int num;

  // Type of activation function.
  activation_type activation;

  dims_t inputs;
  dims_t weights;
  dims_t biases;
  dims_t outputs;

  // A list of sets of weights for this layer.
  //
  // Layers can have different types of weights (e.g. FC synapses, CONV
  // kernels, biases, BN gamma/beta, etc). This list is designed to store each
  // of these types of weights separately, although it is up to the
  // implementation whether to do so or not.  Each entry in the list can be of
  // a different storage type, so some can be compressed and others left
  // uncompressed.
  data_list* host_weights;

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

  // For CONV and POOL layers
  stride_dims stride;

  // CONV layers only.
  padding pad;

  // POOL layers only.
  pool_type pool;

  // Where are the class predictions stored, hid or hid_temp?
  int result_in_temp;

  input_pp input_preprocessing;

  io_req_t input_req;
  io_req_t weights_req;
  io_req_t output_req;
} layer_t;

// A network is a stack of layers and a layer count.
typedef struct _network_t {
  layer_t* layers;
  int depth;
} network_t;

typedef struct _device_t {
    io_req_t cpu_default_offload;
    io_req_t cpu_pooling_offload;
    io_req_t cpu_activation_func_offload;
    data_mvmt_policy weights_load_policy;
    bool use_hw_activation_func;
    bool use_hw_batch_norm;
    bool use_hw_pooling;
    bool use_pipelined_dma;
    bool use_pipelined_activation_func;
    size_t umem_size;
    size_t spad_size;
    size_t l2_size;
    // An implementation can pass any pointer containing architecture specific
    // state that must be shared.
    void* session;
} device_t;

// Sampling parameters to reduce simulation time.
//
// These parameters will allow us to skip work in very regular workload phases.
// How the sampled time is interpreted and upsampled is not handled by SMAUG -
// the user is responsible for doing that.
//
// TODO: These parameters are very backend specific and should be collected
// into backend-specific structures.
typedef struct _sampling_param_t {
    // SMIV: Only run this many output feature maps per conv layer.
    int standard_conv_num_filters;

    // How many neurons should be simulated for an FC layer.
    // Currently UNUSED.
    int fc_num_neurons;

    // SMV: Run this many iterations of the inner tiling loop.
    // See smv/arch/convolution.c for details.
    int smv_conv_inner_iters;

    // SMV: Run this many L2 tiles.
    int smv_conv_l2_tiles;

    // SMV: Run this many input tiles of an L2 tile.
    int smv_conv_input_tiles;

    // SMV: Run this many output tiles for an input tile.
    int smv_conv_output_tiles;
} sampling_param_t;

// Possible values of ARCHITECTURE.
//
// This defines the structure of the nnet accelerator - whether it is a
// monolithic block or a collection of multiple blocks.
//
// Allowed values are: MONOLITHIC, COMPOSABLE, EIGEN, MKLDNN, SMIV, SMV
#define MONOLITHIC 0
#define COMPOSABLE 1
#define SMIV 2
#define EIGEN 3
#define MKLDNN 4
#define SMV 5

// Possible values of SIGMOID_TABLE_IMPL
#define EXP_UNIT 0
#define LUT_CENTERED 1
#define LUT_NONCENTERED 2

// Convert a layer_type enum to a string
#define LAYER_TYPE_STR(arg) \
  (arg == CONV_STANDARD ? "CONV_STANDARD" : \
   arg == CONV_DEPTHWISE ? "CONV_DEPTHWISE" : \
   arg == CONV_POINTWISE ? "CONV_POINTWISE" : \
   arg == POOLING ? "POOLING" : \
   arg == FC ? "FC" : \
   arg == OUTPUT ? "OUTPUT" : \
   arg == INPUT ? "INPUT" : \
   "UNKNOWN")

#define ACTIVATION_TYPE_STR(arg) \
    (arg == NO_ACTIVATION ? "NONE" : \
     arg == RELU ? "RELU" : \
     arg == RELU_THRESHOLD ? "RELU_THRESHOLD" : \
     arg == LRELU ? "LRELU" : \
     arg == ELU ? "ELU" : \
     arg == SELU ? "SELU" : \
     arg == TANH ? "Tanh" : \
     arg == SIGMOID ?  "Sigmoid" : \
     arg == SOFTMAX ? "SOFTMAX" : "UNKNOWN")


#define STRING(arg) #arg

// This is to avoid a ton of spurious unused variable warnings when
// we're not building for gem5.
#define UNUSED(x) (void)(x)

// Convenience macros to switch between invoking an accelerator (if building a
// binary for gem5) or just calling the kernel function in software.
//
// Usage:
//
//  These macros expand differently based on whether the GEM5_HARNESS macro is
//  defined. If so, then this binary is meant to be run under gem5, invoking
//  accelerators; if not, this binary should run the pure software version of
//  the accelerated kernels.
//
//  If GEM5_HARNESS is defined:
//
//     MAP_ARRAY_TO_ACCEL(myReqCode, myArrayName, myArrayPtr, mySize)
//        ===>   mapArrayToAccelerator(myReqCode, myArrayName, myArrayPtr, mySize)
//
//     INVOKE_KERNEL(myReqCode, kernelFuncName, args...)
//        ===>   invokeAcceleratorAndBlock(myReqCode)
//
//  Otherwise:
//     MAP_ARRAY_TO_ACCEL(myReqCode, myArrayName, myArrayPtr, mySize)
//        expands to nothing
//
//     INVOKE_KERNEL(myReqCode, kernelFuncName, args...)
//        ===>  kernelFuncName(args)
//
#ifdef GEM5_HARNESS

#define MAP_ARRAY_TO_ACCEL(req_code, name, base_addr, size)                    \
    mapArrayToAccelerator(req_code, name, base_addr, size)
#define INVOKE_KERNEL(req_code, kernel_ptr, args...)                           \
    do {                                                                       \
        UNUSED(kernel_ptr);                                                    \
        invokeAcceleratorAndBlock(req_code);                                   \
    } while (0)
#define INVOKE_KERNEL_NOBLOCK(req_code, finish_flag, kernel_ptr, args...)      \
    do {                                                                       \
        UNUSED(kernel_ptr);                                                    \
        invokeAcceleratorAndReturn2(req_code, finish_flag);                    \
    } while (0)

#else

#define MAP_ARRAY_TO_ACCEL(req_code, name, base_addr, size)                    \
    do {                                                                       \
        UNUSED(req_code);                                                      \
        UNUSED(name);                                                          \
        UNUSED(base_addr);                                                     \
        UNUSED(size);                                                          \
    } while (0)
#define INVOKE_KERNEL(req_code, kernel_ptr, args...) kernel_ptr(args)
#define INVOKE_KERNEL_NOBLOCK(req_code, finish_flag, kernel_ptr, args...)      \
    kernel_ptr(args)

#endif

// Simplified version of MAP_ARRAY_TO_ACCEL.
//
// This assumes that the current name of the base pointer is also the name of
// the array in the top level function of the dynamic trace. THIS IS VERY
// IMPORTANT - if the argument passed to a top level function has been renamed in
// the function, then this WILL NOT WORK!
//
// MAP_ARRAY(myReqCode, myArray, mySize)
//    ===>   MAP_ARRAY_TO_ACCEL(myReqCode, "myArray", myArray, mySize)
#define MAP_ARRAY(req_code, name_and_base_addr, size)                          \
    MAP_ARRAY_TO_ACCEL(                                                        \
            req_code, STRING(name_and_base_addr), name_and_base_addr, size)


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
#define max2(A, B) (((A) > (B)) ? (A) : (B))
#define max3(e0, e1, e2) max2(max2(e0, e1), e2)
#define max4(e0, e1, e2, e3) max2(max2(e0, e1), max2(e2, e3))
#define max8(e0, e1, e2, e3, e4, e5, e6, e7)                                   \
    max2(max4(e0, e1, e2, e3), max4(e4, e5, e6, e7))
#define max9(e0, e1, e2, e3, e4, e5, e6, e7, e8)                               \
    max2(max8(e0, e1, e2, e3, e4, e5, e6, e7), e8)

#define min2(A, B) (((A) < (B)) ? (A) : (B))

#define FRAC_CEIL(A, B) ((A) / (B) + ((A) % (B) != 0))

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

// Use these convenience macros to cast a raw pointer into a multidimensional
// variable-length array, which lets us use [] notation inside of the ugly
// sub2ind syntax!
//
// Usage:
//   If we have an array like array[5][4]:
//      ARRAY_2D(TYPE, output_name, array, 4);
//
//   If we have an array like array[5][4][3]:
//      ARRAY_3D(TYPE, output_name, array, 4, 3);
//
//   If we have an array like array[5][4][3][2]
//      ARRAY_4D(TYPE, output_name, array, 4, 3, 2);
//
//   And so on...
#define ARRAY_1D(TYPE, output_array_name, input_array_name)                    \
    TYPE* output_array_name = (TYPE*)input_array_name

#define ARRAY_2D(TYPE, output_array_name, input_array_name, DIM_1)             \
    TYPE(*output_array_name)[DIM_1] = (TYPE(*)[DIM_1])input_array_name

#define ARRAY_3D(TYPE, output_array_name, input_array_name, DIM_1, DIM_2)      \
    TYPE(*output_array_name)[DIM_1][DIM_2] =                                   \
        (TYPE(*)[DIM_1][DIM_2])input_array_name

#define ARRAY_4D(                                                              \
    TYPE, output_array_name, input_array_name, DIM_1, DIM_2, DIM_3)            \
        TYPE(*output_array_name)[DIM_1][DIM_2][DIM_3] =                        \
            (TYPE(*)[DIM_1][DIM_2][DIM_3])input_array_name

#define ARRAY_5D(                                                              \
    TYPE, output_array_name, input_array_name, DIM_1, DIM_2, DIM_3, DIM_4)     \
        TYPE(*output_array_name)[DIM_1][DIM_2][DIM_3][DIM_4] =                 \
            (TYPE(*)[DIM_1][DIM_2][DIM_3][DIM_4])input_array_name

#if DEBUG_LEVEL >= 1
  #define INFO_MSG(args...) printf(args)

  #if DEBUG_LEVEL >= 2
    #define PRINT_MSG(args...) printf(args)
    #define PRINT_DEBUG(hid, rows, cols, num_cols)                                 \
        print_debug(hid, rows, cols, num_cols)
    #define PRINT_DEBUG4D(hid, rows, cols, height)                                 \
        print_debug4d(hid, rows, cols, height)
    #define PRINT_DEBUG4D_FP16(hid, num, height, rows, cols)                       \
        print_debug4d_fp16(hid, num, height, rows, cols)

    #if DEBUG_LEVEL >= 3
      #define PRINT_DEBUG_V(hid, rows, cols, num_cols)                               \
          print_debug(hid, rows, cols, num_cols)
      #define PRINT_DEBUG4D_V(hid, rows, cols, height)                               \
          print_debug4d(hid, rows, cols, height)
      #define PRINT_MSG_V(args...) printf(args)
    #else
      #define PRINT_DEBUG_V(hid, rows, cols, num_cols)
      #define PRINT_DEBUG4D_V(hid, rows, cols, height)
      #define PRINT_MSG_V(args...)
    #endif
  #else
    #define PRINT_MSG(args...)
    #define PRINT_DEBUG(hid, rows, cols, num_cols)
    #define PRINT_DEBUG4D(hid, rows, cols, height)
    #define PRINT_DEBUG4D_FP16(hid, num, height, rows, cols)
    #define PRINT_DEBUG_V(hid, rows, cols, height)
    #define PRINT_DEBUG4D_V(hid, rows, cols, height)
    #define PRINT_MSG_V(args...)
  #endif
#else
  #define INFO_MSG(args...)
  #define PRINT_DEBUG(hid, rows, cols, num_cols)
  #define PRINT_DEBUG4D(hid, rows, cols, height)
  #define PRINT_DEBUG4D_FP16(hid, num, height, rows, cols)
  #define PRINT_MSG(args...)
  #define PRINT_DEBUG_V(hid, rows, cols, height)
  #define PRINT_DEBUG4D_V(hid, rows, cols, height)
  #define PRINT_MSG_V(args...)
#endif

#define CACHELINE_SIZE 64
#define LOG_PAGE_SIZE 12

#define ASSERT_MEMALIGN(ptr, err) \
    assert(err == 0 && "Failed to allocate memory for " #ptr ".\n");

// Compiler-specific features.
//
// ALWAYS_INLINE:
// We have to disable all function inlining at the global level for Aladdin +
// LLVM-Tracer to work, but sometimes we do want to force inline functions
// (otherwise we run into all the issues of function call barriers in Aladdin).
// Add ALWAYS_INLINE before the function declaration to force inlining on this
// function.  Don't do this except when we're tracing though; usually it is not
// necessary and it generates a lot of compiler warnings.
//
// ASSERT:
// Disable asserts within instrumented when tracing.
//
// ASSUME_ALIGNED:
// Tell the compiler to assume a pointer is aligned on some byte boundary. This
// is not supported in clang 3.4.
#ifdef TRACE_MODE
#define ALWAYS_INLINE __attribute__((__always_inline__))
#define ASSERT(x)
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#else
#define ALWAYS_INLINE
#define ASSERT(x) assert(x)
#define ASSUME_ALIGNED(ptr, args...) __builtin_assume_aligned((ptr), args)
#endif

#define MAYBE_UNUSED __attribute__((__unused__))

#ifdef GEM5_HARNESS
#define M5_SWITCH_CPU() m5_switch_cpu()
#define M5_RESET_STATS() m5_reset_stats(0, 0)
#define M5_DUMP_STATS() m5_dump_stats(0, 0)
#define M5_DUMP_RESET_STATS() m5_dump_reset_stats(0, 0)
#else
#define M5_SWITCH_CPU()
#define M5_RESET_STATS()
#define M5_DUMP_STATS()
#define M5_DUMP_RESET_STATS()
#endif


//=------------ GLOBAL VARIABLES ---------------=//

extern int NUM_TEST_CASES;
extern int NUM_CLASSES;
extern int INPUT_DIM;
extern float* sigmoid_table;
extern float* exp_table;
extern sigmoid_impl_t SIGMOID_IMPL;

//=------------ --------------------------------=//

#endif
