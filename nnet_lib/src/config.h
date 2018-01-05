#ifndef _CONFIG_H_
#define _CONFIG_H_

#include "core/nnet_fwd_defs.h"

////////////////////////////////////////
/////// USER TUNABLE PARAMETERS ////////
////////////////////////////////////////


// number of stored points in sigmoid lookup table
#define LG_SIGMOID_COARSENESS 4
#define SIG_MIN -5            // lower input bound for sigmoid lookup table
#define SIG_MAX +5            // upper input bound for sigmoid lookup table

// Parameters for optimization
#define NUM_OF_INT_BITS                                                        \
    6  // number of bits before the decimal pt in our representation
#define NUM_OF_FRAC_BITS                                                       \
    26  // number of bits after the decimal pt in our representation

// The architecture of the accelerator
#ifndef ARCHITECTURE
#define ARCHITECTURE MONOLITHIC
#endif

// By default, don't print any debug output.
#ifndef DLEVEL
#define DLEVEL 0
#endif

// Define this to use a table approximation of the sigmoid activation function.
// #define SIGMOID_TABLE

// If 1, then weights for fully-connected layers are stored columnmajor, rather
// than rowmajor, which tends to improve performance when the input data are
// stored as row vectors.
//
// This can also be defined from the build command.
#ifndef TRANSPOSE_WEIGHTS
#define TRANSPOSE_WEIGHTS 0
#endif

// If 1, then this uses a tree-based max implementation for the pooling layers,
// which is more efficient than a loop in HW.
#ifndef TREE_MAX
#define TREE_MAX 0
#endif

// Disable batching for FC layers in SMIV.
#define DISABLE_SMIV_INPUT_BATCHING

// Use the SMIV SIMD implementation.
#define ENABLE_SIMD_IMPL

// Turns out debugging output, which prints out the results of operations.
#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0
#endif

// Print the input data and the complete set of weights.
#ifndef PRINT_DATA_AND_WEIGHTS
#define PRINT_DATA_AND_WEIGHTS 0
#endif

#if ARCHITECTURE == SMIV
#if TRANSPOSE_WEIGHTS == 1
#error "SMIV does not support transposed weights!"
#endif
#define DATA_ALIGNMENT 8
#define PRECOMPUTE_BN_VARIANCE 1

#elif ARCHITECTURE == EIGEN
#define DATA_ALIGNMENT 0
#define PRECOMPUTE_BN_VARIANCE 1

#elif ARCHITECTURE == MKLDNN
#define DATA_ALIGNMENT 0
#define PRECOMPUTE_BN_VARIANCE 0
#if TRANSPOSE_WEIGHTS == 0
#error "MKL-DNN does not support nontransposed weights!"
#endif

#else
#define DATA_ALIGNMENT 0
#define PRECOMPUTE_BN_VARIANCE 1

#endif

/////////////////////////////////////////////////
/////// SHOULD NOT NEED TO CHANGE THESE /////////
/////////////////////////////////////////////////

#if ARCHITECTURE == MONOLITHIC
#define ARCH_STR "MONOLITHIC"
#elif ARCHITECTURE == COMPOSABLE
#define ARCH_STR "COMPOSABLE"
#elif ARCHITECTURE == SMIV
#define ARCH_STR "SMIV"
#elif ARCHITECTURE == EIGEN
#define ARCH_STR "EIGEN"
#elif ARCHITECTURE == MKLDNN
#define ARCH_STR "MKLDNN"
#else
#error "Unknown architecture!"
#endif

#endif
