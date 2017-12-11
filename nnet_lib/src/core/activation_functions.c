#include "utility/utility.h"
#include "nnet_fwd.h"

#include "activation_functions.h"

// Dispatch to the appropriate activation function.
ALWAYS_INLINE
void activation_fun(float* activations,
                    int size,
                    activation_type function,
                    float* sigmoid_table) {
    if (function == RELU) {
        relu(activations, size);
    } else if (function == LRELU) {
        lrelu(activations, size);
    } else if (function == ELU) {
        elu(activations, size);
    } else if (function == SELU) {
        selu(activations, size);
    } else if (function == TANH) {
        tanh_act(activations, size, sigmoid_table);
    } else if (function == SIGMOID) {
        sigmoid_inplace(activations, size, sigmoid_table);
    } else if (function == SOFTMAX) {
        softmax(activations, NUM_TEST_CASES, NUM_CLASSES, sigmoid_table);
    }
}

// The rectified linear activation function
// ** this function is in-place (modifies a) **
ALWAYS_INLINE
void relu(float* a, int num_units) {
    int i;
relu_loop: for (i = 0; i < num_units; i++) {
        if (a[i] < 0.0) {
            a[i] = 0.0;
        }
    }
}

// The leaky rectified linear activation function
// ** this function is in-place (modifies a) **
ALWAYS_INLINE
void lrelu(float* a, int num_units) {
    int i;
    static const float alpha = 0.1;
lrelu_loop:
    for (i = 0; i < num_units; i++) {
        if (a[i] < 0.0) {
            a[i] = alpha * a[i];
        }
    }
}

// The exponential linear activation function
// ** this function is in-place (modifies a) **
ALWAYS_INLINE
void elu(float* a, int num_units) {
    int i;
    static const float alpha = 1.0;
elu_loop:
    for (i = 0; i < num_units; i++) {
        if (a[i] < 0.0) {
            a[i] = alpha * (exp(a[i]) - 1);
        }
    }
}

// The scaled exponential linear activation function
// ** this function is in-place (modifies a) **
ALWAYS_INLINE
void selu(float* a, int num_units) {
    int i;
    static const float lamda = 1.0;

    elu(a, num_units);
selu_loop:
    for (i = 0; i < num_units; i++) {
        a[i] = lamda * a[i];
    }
}

// The hyberbolic sine activation function
// ** this function is in-place (modifies a) **
ALWAYS_INLINE
void tanh_act(float* a, int num_units, float* sigmoid_table) {
    int i;
tanh_act_loop1:
    for (i = 0; i < num_units; i++) {
        a[i] = 2 * a[i];
    }
    sigmoid_inplace(a, num_units, sigmoid_table);

tanh_act_loop2:
   for (i = 0; i < num_units; i++) {
        a[i] = 2 * a[i] - 1;
    }
}

// The logistic activation function.
//
// Operates on a single float.
float sigmoid(float a) {
    return 1.0 / (1.0 + exp(-a));
}


ALWAYS_INLINE
void sigmoid_inplace(float* a, int num_units, float* sigmoid_table) {
#ifdef SIGMOID_TABLE
  sigmoid_lookup(a, num_units, sigmoid_table);
#else
  sigmoidn(a, num_units);
#endif
}

// The logistic activation function
// ** this function is in-place (modifies a) **
void sigmoidn(float* a, int num_units) {
    int i;
    float value;
sigmoidn_loop: for (i = 0; i < num_units; i++) {
        value = 1.0 / (1.0 + exp(-a[i]));
        a[i] = conv_float2fixed(value);
    }
}

// The logistic activation function, implemented with a lookup table
// and linear interpolation
// ** this function is in-place (modifies a) **
void sigmoid_lookup(float* a, int num_units, float* sigmoid_table) {
    int i, ind;
    float temp, delta_x;
    float SIG_RANGE = SIG_MAX - SIG_MIN;
    sigmoid_table_loop:
    for (i = 0; i < num_units; i++) {
        if (a[i] < SIG_MIN) {
            a[i] = 0.0;  // do I need to convert these?? I guess not?
        } else if (a[i] >= SIG_MAX) {
            a[i] = 1.0;
        } else {
            temp = conv_float2fixed(((a[i] - SIG_MIN) / SIG_RANGE) *
                                    ((1 << LG_SIGMOID_COARSENESS) - 1.0));
            ind = (int)temp;
            delta_x = conv_float2fixed(temp - ind);  // in [0,1]
            // printf("%f   %f\n", delta_x, sigmoid_table[ind]);
            a[i] = conv_float2fixed(sigmoid_table[ind] * (1.0 - delta_x) +
                                    sigmoid_table[ind + 1] * delta_x);
        }
    }
}

// Softmax function on matrix a
// a is num_test_cases by num_classes
// the softmax function exponentiates each element and then normalizes each row
// to sum to 1
// ** this function is in-place (modifies a) **
void softmax(float* a,
               int num_test_cases,
               int num_classes,
               float* sigmoid_table) {
    ARRAY_2D(float, _a, a, num_classes);
    int i, j;
    float normaliz;

    sigmoid_inplace(a, num_test_cases * num_classes, sigmoid_table);
softmax_outer: for (i = 0; i < num_test_cases; i++) {
        // compute the normalization factor
        normaliz = 0.0;
softmax_inner0: for (j = 0; j < num_classes; j++) {
            normaliz += _a[i][j];
        }
softmax_inner1: for (j = 0; j < num_classes; j++) {
            _a[i][j] /= normaliz;
        }
    }
}

