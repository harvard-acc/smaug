#ifndef _OPERATORS_SMV_KERNELS_ACTIVATION_FUNCTIONS_SIMD_H_
#define _OPERATORS_SMV_KERNELS_ACTIVATION_FUNCTIONS_SIMD_H_

#include "assert.h"
#include "stdio.h"

#include "smaug/operators/common.h"
#include "smaug/operators/smv/kernels/params.h"

#ifdef __cplusplus
extern "C" {
#endif

// The rectified linear activation function
ALWAYS_INLINE
static inline v8fp_t relu_vec_unit(v8fp_t a) {
    v8fp_t zero = (v8fp_t){ 0 };
    v8sfx_t mask = (a > zero);
    return VEC256_MASK(a, mask);
}

ALWAYS_INLINE
static inline void relu_vec(v8fp_t* inputs, v8fp_t* results, int inputs_size) {
    relu_loop:
    for (int i = 0; i < inputs_size / VECTOR_SIZE; i++) {
        results[i] = relu_vec_unit(inputs[i]);
    }
}

// The leaky rectified linear activation function
ALWAYS_INLINE
static inline v8fp_t lrelu_vec_unit(v8fp_t a, float slope) {
    v8fp_t zero = (v8fp_t){ 0 };
    v8fp_t slope_vec =
            (v8fp_t){ slope, slope, slope, slope, slope, slope, slope, slope };
    v8sfx_t neg_mask = a < zero;
    v8sfx_t pos_mask = a >= zero;
    v8fp_t scaled = slope_vec * a;
    v8fp_t first = VEC256_MASK(scaled, neg_mask);
    v8fp_t second = VEC256_MASK(a, pos_mask);
    return VEC256_MASK(scaled, neg_mask) + VEC256_MASK(a, pos_mask);
}

ALWAYS_INLINE
static inline void lrelu_vec(v8fp_t* inputs,
                             v8fp_t* results,
                             int inputs_size,
                             float slope) {
    lrelu_loop:
    for (int i = 0; i < inputs_size / VECTOR_SIZE; i++) {
        results[i] = lrelu_vec_unit(inputs[i], slope);
    }
}

// The exponential linear activation function
ALWAYS_INLINE
static inline v8fp_t elu_vec_unit(v8fp_t a, float alpha) {
    elu_unit_loop:
    for (int i = 0; i < VECTOR_SIZE; i++) {
        float value = a[i];
        if (value < 0.0) {
            a[i] = alpha * (exp(value) - 1);
        }
    }
    return a;
}

ALWAYS_INLINE
static inline void elu_vec(v8fp_t* inputs,
                           v8fp_t* results,
                           int inputs_size,
                           float alpha) {
    elu_loop:
    for (int i = 0; i < inputs_size / VECTOR_SIZE; i++) {
        results[i] = elu_vec_unit(inputs[i], alpha);
    }
}

// The scaled exponential linear activation function
ALWAYS_INLINE
static inline v8fp_t selu_vec_unit(v8fp_t a, float alpha, float lambda) {
    a = elu_vec_unit(a, alpha);
    selu_unit_loop:
    for (int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = lambda * a[i];
    }
    return a;
}

ALWAYS_INLINE
static inline void selu_vec(v8fp_t* inputs,
                            v8fp_t* results,
                            int inputs_size,
                            float alpha,
                            float lambda) {
    selu_loop:
    for (int i = 0; i < inputs_size / VECTOR_SIZE; i++) {
        results[i] = selu_vec_unit(inputs[i], alpha, lambda);
    }
}

// The logistic activation function
ALWAYS_INLINE
static inline v8fp_t sigmoid_vec_unit(v8fp_t a) {
    int i;
    float value;
    sigmoid_unit_loop:
    for (i = 0; i < VECTOR_SIZE; i++) {
        a[i] = 1.0 / (1.0 + exp(-a[i]));
    }
    return a;
}

ALWAYS_INLINE
static inline void sigmoid_vec(v8fp_t* inputs,
                               v8fp_t* results,
                               int inputs_size) {
    sigmoid_loop:
    for (int i = 0; i < inputs_size / VECTOR_SIZE; i++) {
        results[i] = sigmoid_vec_unit(inputs[i]);
    }
}

// The hyberbolic sine activation function
ALWAYS_INLINE
static inline v8fp_t tanh_vec_unit(v8fp_t a) {
    v8fp_t one = { 1, 1, 1, 1, 1, 1, 1, 1 };
    v8fp_t two = { 2, 2, 2, 2, 2, 2, 2, 2 };
    v8fp_t two_a = two * a;
    v8fp_t sig = sigmoid_vec_unit(two_a);
    return two * sig - one;
}

ALWAYS_INLINE
static inline void tanh_vec(v8fp_t* inputs, v8fp_t* results, int inputs_size) {
    tanh_loop:
    for (int i = 0; i < inputs_size / VECTOR_SIZE; i++) {
        results[i] = tanh_vec_unit(inputs[i]);
    }
}

// The hard hyberbolic sine activation function
ALWAYS_INLINE
static inline v8fp_t hard_tanh_vec_unit(v8fp_t a, float min, float max) {
    hard_tanh_unit_loop:
    for (int i = 0; i < VECTOR_SIZE; i++) {
        float value = a[i];
        a[i] = value < min ? min : value > max ? max : value;
    }
    return a;
}

ALWAYS_INLINE
static inline void hard_tanh_vec(v8fp_t* inputs,
                                 v8fp_t* results,
                                 int inputs_size,
                                 float min,
                                 float max) {
    hard_tanh_loop:
    for (int i = 0; i < inputs_size / VECTOR_SIZE; i++) {
        results[i] = hard_tanh_vec_unit(inputs[i], min, max);
    }
}

ALWAYS_INLINE
static inline void activation_fun_vec(float* inputs,
                                      float* results,
                                      int inputs_size,
                                      activation_type function,
                                      activation_param_t params) {
    VEC_ARRAY_1D(v8fp_t, _inputs, inputs);
    VEC_ARRAY_1D(v8fp_t, _results, results);
    if (function == RELU) {
        relu_vec(_inputs, _results, inputs_size);
    } else if (function == LRELU) {
        lrelu_vec(_inputs, _results, inputs_size, params.slope);
    } else if (function == ELU) {
        elu_vec(_inputs, _results, inputs_size, params.alpha);
    } else if (function == SELU) {
        selu_vec(_inputs, _results, inputs_size, params.alpha, params.lambda);
    } else if (function == TANH) {
        tanh_vec(_inputs, _results, inputs_size);
    } else if (function == HARD_TANH) {
        hard_tanh_vec(_inputs, _results, inputs_size, params.min, params.max);
    } else if (function == SIGMOID) {
        sigmoid_vec(_inputs, _results, inputs_size);
    } else if (function == SOFTMAX) {
        assert(false && "Softmax SIMD shouldn't be called from here!");
    }
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
