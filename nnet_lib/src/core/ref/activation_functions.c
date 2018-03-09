#include <float.h>

#include "core/ref/activation_functions.h"
#include "core/ref/lookup_tables_ops.h"
#include "utility/utility.h"
#include "nnet_fwd.h"

// The logistic activation function.
//
// Operates on a single float.
ALWAYS_INLINE
float sigmoid_fxp(float a) {
    return 1.0 / (1.0 + exp(-a));
}

ALWAYS_INLINE
void activation_fun_fxp(float* activations,
                        int batch_size,
                        int input_size,
                        int input_pad,
                        activation_type function) {
    int total_size = input_size * batch_size;
    if (function == RELU) {
        relu(activations, total_size);
    } else if (function == LRELU) {
        static const float alpha = 0.1;
        lrelu(activations, total_size, alpha);
    } else if (function == ELU) {
        static const float alpha = 0.1;
        elu(activations, total_size, alpha, activations);
    } else if (function == SELU) {
        selu(activations, total_size);
    } else if (function == TANH) {
        tanh_act(activations, total_size, activations);
    } else if (function == SIGMOID) {
        sigmoid_inplace(activations, total_size);
    } else if (function == SOFTMAX) {
        softmax(activations, batch_size, input_size, input_pad);
    }
}

// Dispatch to the appropriate activation function.
ALWAYS_INLINE
void activation_fun(float* activations,
                    int batch_size,
                    int input_size,
                    int input_pad,
                    activation_type function) {
    activation_fun_fxp(
            activations, batch_size, input_size, input_pad, function);
}

// The rectified linear activation function
// ** this function is in-place (modifies a) **
ALWAYS_INLINE
void relu(float* a, int num_units) {
    int i;
    relu_loop:
    for (i = 0; i < num_units; i++) {
        if (a[i] < 0.0) {
            a[i] = 0.0;
        }
    }
}

// The leaky rectified linear activation function
// ** this function is in-place (modifies a) **
ALWAYS_INLINE
void lrelu(float* a, int num_units, float alpha) {
    int i;
    lrelu_loop:
    for (i = 0; i < num_units; i++) {
        if (a[i] < 0.0) {
            a[i] = alpha * a[i];
        }
    }
}

void elu_expunit(float* a, int num_units, float alpha, float* results) {
    elu_loop:
    for (int i = 0; i < num_units; i++) {
        float value = a[i];
        if (value < 0.0) {
            results[i] = alpha * (exp(value) - 1);
        } else {
            results[i] = value;
        }
    }
}

ALWAYS_INLINE
void elu_lut_fxp(float* a, int num_units, float alpha, float* results) {
    elu_loop:
    for (int i = 0; i < num_units; i++) {
        float value = a[i];
        if (value < 0.0) {
            results[i] = alpha * (exp_lut_fxp(value) - 1);
        } else {
            results[i] = value;
        }
    }
}

// The exponential linear activation function
// ** this function is in-place (modifies a) **
ALWAYS_INLINE
void elu(float* a, int num_units, float alpha, float* results) {
    if (SIGMOID_IMPL == ExpUnit) {
        elu_expunit(a, num_units, alpha, results);
    } else {
        elu_lut_fxp(a, num_units, alpha, results);
    }
}

// The scaled exponential linear activation function
// ** this function is in-place (modifies a) **
ALWAYS_INLINE
void selu(float* a, int num_units) {
    int i;
    static const float alpha = 1.6733;
    static const float lamda = 1.0507;

    elu(a, num_units, alpha, a);
    selu_loop:
    for (i = 0; i < num_units; i++) {
        a[i] = lamda * a[i];
    }
}

// The hyberbolic sine activation function
ALWAYS_INLINE
void tanh_act(float* a, int num_units, float* results) {
    int i;
    tanh_act_loop1:
    for (i = 0; i < num_units; i++) {
        results[i] = 2 * a[i];
    }
    sigmoid_inplace(results, num_units);

    tanh_act_loop2:
    for (i = 0; i < num_units; i++) {
        results[i] = 2 * results[i] - 1;
    }
}


ALWAYS_INLINE
void sigmoid_inplace(float* a, int num_units) {
    if (SIGMOID_IMPL == EXP_UNIT) {
        sigmoidn(a, num_units);
    } else if (SIGMOID_IMPL == CenteredLUT) {
        sigmoid_lookup_centered(a, num_units, a);
    } else if (SIGMOID_IMPL == NoncenteredLUT) {
        sigmoid_lookup_noncentered(a, num_units, a);
    }
}

// The logistic activation function
// ** this function is in-place (modifies a) **
ALWAYS_INLINE
void sigmoidn(float* a, int num_units) {
    int i;
    float value;
    sigmoidn_loop:
    for (i = 0; i < num_units; i++) {
        value = sigmoid_fxp(a[i]);
        a[i] = conv_float2fixed(value);
    }
}

ALWAYS_INLINE
void sigmoid_lookup_centered(float* a, int num_units, float* results) {
    sigmoid_loop:
    for (int i = 0; i < num_units; i++) {
        results[i] = sigmoid_lookup_centered_op_fxp(a[i]);
    }
}

ALWAYS_INLINE
void sigmoid_lookup_noncentered(float* a, int num_units, float* results) {
    sigmoid_loop:
    for (int i = 0; i < num_units; i++) {
        results[i] = sigmoid_lookup_noncentered_op_fxp(a[i]);
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
             int input_pad) {
    ARRAY_2D(float, _a, a, softmax_size + input_pad);

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

