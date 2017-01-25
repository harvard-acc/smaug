#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// Fixed parameters
#define INPUT_DIM 784
#define NUM_CLASSES 10
#define LG_SIGMOID_COARSENESS 4 // number of stored points in sigmoid lookup table
#define NUM_TEST_CASES 100 // NOT READ BY nnet_fwd.c, ONLY BY the other one
#define SIG_MIN -5      // lower input bound for sigmoid lookup table
#define SIG_MAX +5      // upper input bound for sigmoid lookup table

// Parameters for optimization
#define NUM_LAYERS 1
int NUM_HIDDEN_UNITS[NUM_LAYERS] = {
  5
};
#define ACTIVATION_FUN 0    // categorical, 0=RELU 1=sigmoid lookup 2=true sigmoid
#define NUM_OF_INT_BITS 6  // number of bits before the decimal pt in our representation
#define NUM_OF_FRAC_BITS 26 // number of bits after the decimal pt in our representation

#define PRINT_DEBUG 0

#ifndef BITWIDTH_QUANTIZATION
#define conv_float2fixed(X) (X)
#endif

#define INPUTS_FILENAME "/home/jmh/projects/pesc_hardware/HardwareNets/../mnist/validation_data_textual_all_10000.txt"
#define LABELS_FILENAME "/home/jmh/projects/pesc_hardware/HardwareNets/../mnist/validation_labels_textual_all_10000_not_one_hot.txt"
#define WEIGHTS_FILENAME "/home/jmh/projects/pesc_hardware/HardwareNets/../mnist/mnist_textual_weights.txt"
