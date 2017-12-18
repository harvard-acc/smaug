#include <float.h>

#include "utility/utility.h"
#include "nnet_fwd.h"

#include "activation_functions.h"

// Dispatch to the appropriate activation function.
ALWAYS_INLINE
void activation_fun(float* activations,
                    int batch_size,
                    int input_size,
                    activation_type function,
                    float* sigmoid_table) {
    int total_size = input_size * batch_size;
    if (function == RELU) {
        relu(activations, total_size);
    } else if (function == LRELU) {
        lrelu(activations, total_size);
    } else if (function == ELU) {
        elu(activations, total_size);
    } else if (function == SELU) {
        selu(activations, total_size);
    } else if (function == TANH) {
        tanh_act(activations, total_size, sigmoid_table);
    } else if (function == SIGMOID) {
        sigmoid_inplace(activations, total_size, sigmoid_table);
    } else if (function == SOFTMAX) {
        softmax(activations, batch_size, input_size, sigmoid_table);
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

// Softmax function on matrix a.
//
// The softmax function exponentiates each element and then normalizes each row
// to sum to 1.
//
// Args:
//   a: Matrix of size num_test_cases x softmax_size, stored rowmajor. This
//      contains both inputs and the outputs.
//   num_test_cases: batch size.
//   softmax_size: number of activations per input.
//   sigmoid_table: Table for lookup based sigmoid (unused right now).
//
// To improve numerical stability, we use the max trick: all elements are first
// subtracted by the maximum value in each input before being exponentiated.
//
// This function is in-place (modifies a).
void softmax(float* a,
             int num_test_cases,
             int softmax_size,
             float* sigmoid_table) {
    ARRAY_2D(float, _a, a, softmax_size);

    // Compute the maximum of the elements in groups of 8 and the remainder one
    // by one.
    int max8_remainder = softmax_size - ((softmax_size >> 3) << 3);

    softmax_batch:
    for (int i = 0; i < num_test_cases; i++) {
        // Find the maximum of each input.
        float max_elem = -FLT_MAX;
        softmax_max_loop0:
        for (int j = 0; j < softmax_size - max8_remainder; j += 8) {
            max_elem = max9(max_elem,
                            _a[i][j],
                            _a[i][j + 1],
                            _a[i][j + 2],
                            _a[i][j + 3],
                            _a[i][j + 4],
                            _a[i][j + 5],
                            _a[i][j + 6],
                            _a[i][j + 7]);
        }
        // Do the remainder.
        softmax_max_loop1:
        for (int j = softmax_size - max8_remainder - 1; j < softmax_size; j++) {
            max_elem = max2(max_elem, _a[i][j]);
        }

        // Subtract the max from each activation.
        softmax_max_sub:
        for (int j = 0; j < softmax_size; j++) {
            _a[i][j] -= max_elem;
        }

        // Now exponentiate.
        softmax_exp:
        for (int j =0; j < softmax_size; j++) {
            _a[i][j] = exp(_a[i][j]);
        }

        // Compute the normalization factor, separately from the exponentiation,
        // making it easier for Aladdin to turn this into an adder tree.
        float normaliz = 0.0;
        softmax_inner0:
        for (int j = 0; j < softmax_size; j++) {
            normaliz += _a[i][j];
        }
        // Precompute the division so that later we can just do a multiplication.
        normaliz = 1.0 / (normaliz + 1e-6);  // epsilon for numerical stability.

        softmax_inner1:
        for (int j = 0; j < softmax_size; j++) {
            _a[i][j] *= normaliz;
        }
    }
}

