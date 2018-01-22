#include <assert.h>
#include <float.h>

#include "core/ref/activation_functions.h"
#include "core/smiv/activation_functions_simd.h"
#include "core/smiv/params.h"
#include "utility/utility.h"
#include "nnet_fwd.h"

// The logistic activation function.
//
// Operates on a single vector of floats.

ALWAYS_INLINE
v8fp_t activation_fun_simd_fxp(v8fp_t activations, activation_type function) {
    if (function == RELU) {
        return relu_simd(activations);
    } else if (function == LRELU) {
        return lrelu_simd(activations);
    } else if (function == ELU) {
        return elu_simd(activations, 0.1);
    } else if (function == SELU) {
        return selu_simd(activations);
    } else if (function == TANH) {
        return tanh_act_simd(activations);
    } else if (function == SIGMOID) {
        return sigmoid_inplace_simd(activations);
    } else if (function == SOFTMAX) {
        assert(false && "Softmax SIMD not supported!");
    }
    return activations;
}

// Dispatch to the appropriate activation function.
ALWAYS_INLINE
v8fp_t activation_fun_simd(v8fp_t activations, activation_type function) {
    return activation_fun_simd_fxp(activations, function);
}

// The rectified linear activation function
// ** this function is in-place (modifies a) **
ALWAYS_INLINE
v8fp_t relu_simd(v8fp_t a) {
    v8fp_t zero = (v8fp_t){ 0 };
    v8sfx_t mask = (a > zero);
    return VEC_MASK(a, mask);
}

// The leaky rectified linear activation function
// ** this function is in-place (modifies a) **
ALWAYS_INLINE
v8fp_t lrelu_simd(v8fp_t a) {
    static const float alpha = 0.1;
    v8fp_t zero = (v8fp_t){ 0 };
    v8fp_t alpha_vec =
            (v8fp_t){ alpha, alpha, alpha, alpha, alpha, alpha, alpha, alpha };
    v8sfx_t neg_mask = a < zero;
    v8sfx_t pos_mask = a >= zero;
    v8fp_t scaled = alpha_vec * a;
    return VEC_MASK(scaled, neg_mask) + VEC_MASK(a, pos_mask);
}

ALWAYS_INLINE
v8fp_t elu_expunit_simd(v8fp_t a, float alpha) {
    elu_loop:
    for (int i = 0; i < VECTOR_SIZE; i++) {
        float value = a[i];
        if (value < 0.0) {
            a[i] = alpha * (exp(value) - 1);
        }
    }
    return a;
}

ALWAYS_INLINE
v8fp_t elu_lut_simd(v8fp_t a, float alpha) {
    elu_loop:
    for (int i = 0; i < VECTOR_SIZE; i++) {
        float value = a[i];
        if (value < 0.0) {
            a[i] = alpha * (exp_lut(value) - 1);
        }
    }
    return a;
}

// The exponential linear activation function
// ** this function is in-place (modifies a) **
ALWAYS_INLINE
v8fp_t elu_simd(v8fp_t a, float alpha) {
    if (SIGMOID_IMPL == ExpUnit) {
        return elu_expunit_simd(a, alpha);
    } else {
        return elu_lut_simd(a, alpha);
    }
}

// The scaled exponential linear activation function
// ** this function is in-place (modifies a) **
ALWAYS_INLINE
v8fp_t selu_simd(v8fp_t a) {
    int i;
    static const float alpha = 1.6733;
    static const float lamda = 1.0507;

    a = elu_simd(a, alpha);
    selu_loop:
    for (i = 0; i < VECTOR_SIZE; i++) {
        a[i] = lamda * a[i];
    }
    return a;
}

// The hyberbolic sine activation function
// ** this function is in-place (modifies a) **
ALWAYS_INLINE
v8fp_t tanh_act_simd(v8fp_t a) {
    v8fp_t one = { 1, 1, 1, 1, 1, 1, 1, 1 };
    v8fp_t two = { 2, 2, 2, 2, 2, 2, 2, 2 };
    v8fp_t two_a = two * a;
    v8fp_t sig = sigmoid_inplace_simd(two_a);
    return two * sig - one;
}


ALWAYS_INLINE
v8fp_t sigmoid_inplace_simd(v8fp_t a) {
    if (SIGMOID_IMPL == EXP_UNIT) {
        return sigmoidn_simd(a);
    } else if (SIGMOID_IMPL == CenteredLUT) {
        return sigmoid_lookup_centered_simd(a);
    } else if (SIGMOID_IMPL == NoncenteredLUT) {
        return sigmoid_lookup_noncentered_simd(a);
    }
    assert(false && "Unknown SIGMOID implementation type!");
    return (v8fp_t){ 0 };
}

// The logistic activation function
// ** this function is in-place (modifies a) **
v8fp_t sigmoidn_simd(v8fp_t a) {
    int i;
    float value;
    sigmoidn_loop:
    for (i = 0; i < VECTOR_SIZE; i++) {
        value = sigmoid(a[i]);
        a[i] = conv_float2fixed(value);
    }
    return a;
}

ALWAYS_INLINE
v8fp_t sigmoid_lookup_centered_simd(v8fp_t a) {
    sigmoid_loop:
    for (int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = sigmoid_lookup_centered_op_fxp(a[i]);
    }
    return a;
}

ALWAYS_INLINE
v8fp_t sigmoid_lookup_noncentered_simd(v8fp_t a) {
    sigmoid_loop:
    for (int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = sigmoid_lookup_noncentered_op_fxp(a[i]);
    }
    return a;
}
