#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// Fixed parameters
#define INPUT_DIM 784
#define NUM_CLASSES 10
#define LG_SIGMOID_COARSENESS 4 // number of stored points in sigmoid lookup table
#define NUM_TEST_CASES 1 // NOT READ BY nnet_fwd.c, ONLY BY the other one
#define SIG_MIN -5      // lower input bound for sigmoid lookup table
#define SIG_MAX +5      // upper input bound for sigmoid lookup table

// Parameters for optimization
#define NUM_LAYERS 3
int NUM_HIDDEN_UNITS[NUM_LAYERS] = {
  32,32,32
}; 
#define ACTIVATION_FUN 0    // categorical, 0=RELU 1=sigmoid lookup 2=true sigmoid
#define NUM_OF_INT_BITS 11  // number of bits before the decimal pt in our representation
#define NUM_OF_FRAC_BITS 21 // number of bits after the decimal pt in our representation

#define PRINT_DEBUG 0

#define INPUTS_FILENAME "<none>"
#define LABELS_FILENAME "<none>"
#define WEIGHTS_FILENAME "<none>"
