#ifndef _NNET_FWD_H_
#define _NNET_FWD_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Fixed parameters
#define INPUT_DIM 10
#define NUM_CLASSES 10
// number of stored points in sigmoid lookup table
#define LG_SIGMOID_COARSENESS 4
#define NUM_TEST_CASES 100    // NOT READ BY nnet_fwd.c, ONLY BY the other one
#define SIG_MIN -5            // lower input bound for sigmoid lookup table
#define SIG_MAX +5            // upper input bound for sigmoid lookup table

// Parameters for optimization
#define NUM_LAYERS 1

#define ACTIVATION_FUN 0  // categorical, 0=RELU 1=sigmoid lookup 2=true sigmoid
#define NUM_OF_INT_BITS                                                        \
    6  // number of bits before the decimal pt in our representation
#define NUM_OF_FRAC_BITS                                                       \
    26  // number of bits after the decimal pt in our representation

#define DEBUG 0

#define INPUTS_FILENAME                                                        \
    "/home/jmh/projects/pesc_hardware/HardwareNets/../mnist/"                  \
    "validation_data_textual_all_10000.txt"
#define LABELS_FILENAME                                                        \
    "/home/jmh/projects/pesc_hardware/HardwareNets/../mnist/"                  \
    "validation_labels_textual_all_10000_not_one_hot.txt"
#define WEIGHTS_FILENAME                                                       \
    "/home/jmh/projects/pesc_hardware/HardwareNets/../mnist/"                  \
    "mnist_textual_weights.txt"

#define sub2ind(r, c, n_columns) r* n_columns + c

#if DEBUG == 1
#define PRINT_DEBUG(hid, rows, cols, num_cols)                                 \
    print_debug(hid, rows, col, num_cols)
#else
#define PRINT_DEBUG(hid, rows, cols, num_cols)
#endif

#define CACHELINE_SIZE 32

#define ASSERT_MEMALIGN(ptr, err) \
    assert(err == 0 && "Failed to allocate memory for " #ptr ".\n");

#endif
