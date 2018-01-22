#include <float.h>

#include "core/ref/activation_functions.h"
#include "utility/utility.h"
#include "nnet_fwd.h"

// The logistic activation function.
//
// Operates on a single float.
ALWAYS_INLINE
float sigmoid(float a) {
    return 1.0 / (1.0 + exp(-a));
}

// Build the sigmoid lookup table.
//
// We can either build a centered table, storing values from -5 to 5, or a
// non-centered table, storing values from 0 to 5 (with -5 to 0 calculated by
// subtracting 1). The non-centered table can store the sigmoid with twice the
// precision and does not require the linear interpolation used by the centered
// table.
void init_sigmoid_table(float** table_ptr) {
    if (SIGMOID_IMPL == ExpUnit) {
        *table_ptr = NULL;
        return;
    }

    PRINT_MSG("Initializing sigmoid lookup table.\n");
    *table_ptr = (float*)malloc_aligned(SIG_TABLE_SIZE * sizeof(float));
    float sig_step = 0, x_sig = 0;
    if (SIGMOID_IMPL == CenteredLUT) {
        sig_step = (float)(SIG_RANGE) / (SIG_TABLE_SIZE - 1.0);
        x_sig = (float)SIG_MIN;
    } else if (SIGMOID_IMPL == NoncenteredLUT) {
        sig_step = (float)(SIG_MAX) / (SIG_TABLE_SIZE - 1.0);
        x_sig = 0;
    }

    for (int i = 0; i < SIG_TABLE_SIZE; i++) {
        (*table_ptr)[i] = conv_float2fixed(sigmoid(x_sig));
        // printf("%f, %f\n", x_sig, (*table_ptr)[i]);
        x_sig += sig_step;
    }
}

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
        elu(activations, total_size, 0.1);
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
void elu(float* a, int num_units, float alpha) {
    int i;
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
    static const float alpha = 1.6733;
    static const float lamda = 1.0507;

    elu(a, num_units, alpha);
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


ALWAYS_INLINE
void sigmoid_inplace(float* a, int num_units, float* sigmoid_table) {
    if (SIGMOID_IMPL == EXP_UNIT) {
        sigmoidn(a, num_units);
    } else if (SIGMOID_IMPL == CenteredLUT) {
        sigmoid_lookup_centered(a, num_units, sigmoid_table);
    } else if (SIGMOID_IMPL == NoncenteredLUT) {
        sigmoid_lookup_noncentered(a, num_units, sigmoid_table);
    }
}

// The logistic activation function
// ** this function is in-place (modifies a) **
void sigmoidn(float* a, int num_units) {
    int i;
    float value;
    sigmoidn_loop:
    for (i = 0; i < num_units; i++) {
        value = sigmoid(a[i]);
        a[i] = conv_float2fixed(value);
    }
}

// The logistic activation function, implemented with a lookup table
// and linear interpolation
// ** this function is in-place (modifies a) **
ALWAYS_INLINE
float sigmoid_lookup_centered_op(float a, float* sigmoid_table) {
    float result;
    if (a < SIG_MIN) {
        result = 0.0;  // do I need to convert these?? I guess not?
    } else if (a >= SIG_MAX) {
        result = 1.0;
    } else {
        float temp = conv_float2fixed(((a - SIG_MIN) / SIG_RANGE) *
                                      (SIG_TABLE_SIZE - 1.0));
        int ind = (int)temp;
        float delta_x = conv_float2fixed(temp - ind);  // in [0,1]
        result = conv_float2fixed(sigmoid_table[ind] * (1.0 - delta_x) +
                                  sigmoid_table[ind + 1] * delta_x);
    }
    return result;
}

// The logistic activation function, implemented with a lookup table.
//
// This assumes the sigmoid table is built with positive values of x only
// (since the function is symmetric about x=0, y=1).
// ** this function is in-place (modifies a) **
ALWAYS_INLINE
float sigmoid_lookup_noncentered_op(float a, float* sigmoid_table) {
    float abs_val = a >= 0 ? a : -a;
    float result;
    if (abs_val > SIG_MAX) {
        result = 1.0;
    } else {
        float temp = conv_float2fixed((abs_val * (1.0 / SIG_MAX)) *
                                      (SIG_TABLE_SIZE - 1.0));
        int ind = (int)temp;  // Ideally would be a proper rounding.
        result = conv_float2fixed(sigmoid_table[ind]);
    }
    if (a < 0)
        result = 1.0 - result;
    return result;
}

void sigmoid_lookup_centered(float* a, int num_units, float* sigmoid_table) {
    sigmoid_loop:
    for (int i = 0; i < num_units; i++) {
        a[i] = sigmoid_lookup_centered_op(a[i], sigmoid_table);
    }
}

void sigmoid_lookup_noncentered(float* a, int num_units, float* sigmoid_table) {
    sigmoid_loop:
    for (int i = 0; i < num_units; i++) {
        a[i] = sigmoid_lookup_noncentered_op(a[i], sigmoid_table);
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

