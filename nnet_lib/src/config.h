#ifndef _CONFIG_H_
#define _CONFIG_H_

////////////////////////////////////////
/////// USER TUNABLE PARAMETERS ////////
////////////////////////////////////////


// number of stored points in sigmoid lookup table
#define LG_SIGMOID_COARSENESS 4
#define SIG_MIN -5            // lower input bound for sigmoid lookup table
#define SIG_MAX +5            // upper input bound for sigmoid lookup table

// Parameters for optimization
#define ACTIVATION_FUN 0  // categorical, 0=RELU 1=sigmoid lookup 2=true sigmoid
#define NUM_OF_INT_BITS                                                        \
    6  // number of bits before the decimal pt in our representation
#define NUM_OF_FRAC_BITS                                                       \
    26  // number of bits after the decimal pt in our representation

// The architecture of the accelerator
#ifndef ARCHITECTURE
#define ARCHITECTURE COMPOSABLE
#endif

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

#endif
