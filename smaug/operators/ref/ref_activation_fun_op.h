#ifndef _OPERATORS_REF_ACTIVATION_FUN_OP_H_
#define _OPERATORS_REF_ACTIVATION_FUN_OP_H_

/** \ingroup AladdinKernels
 * @{
 */

#include "assert.h"
#include "stdio.h"
#include "math.h"

#include "smaug/operators/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// All the activation function implementations need to live in here for function
// inlining.

ALWAYS_INLINE
static inline void relu(float* inputs, float* results, int input_size) {
    relu_loop:
    for (int i = 0; i < input_size; i++) {
        float value = inputs[i];
        if (value < 0.0) {
            results[i] = 0.0;
        } else {
            results[i] = value;
        }
    }
}

ALWAYS_INLINE
static inline void lrelu(float* inputs,
                         float* results,
                         int input_size,
                         float slope) {
    lrelu_loop:
    for (int i = 0; i < input_size; i++) {
        float value = inputs[i];
        if (value < 0.0) {
            results[i] = slope * value;
        } else {
            results[i] = value;
        }
    }
}

ALWAYS_INLINE
static inline void elu(float* inputs,
                       float* results,
                       int input_size,
                       float alpha) {
    elu_loop:
    for (int i = 0; i < input_size; i++) {
        float value = inputs[i];
        if (value < 0.0) {
            results[i] = alpha * (exp(value) - 1);
        } else {
            results[i] = value;
        }
    }
}

ALWAYS_INLINE
static inline void selu(float* inputs,
                        float* results,
                        int input_size,
                        float alpha,
                        float lambda) {
    elu(inputs, results, input_size, alpha);
    selu_loop:
    for (int i = 0; i < input_size; i++) {
        results[i] = lambda * results[i];
    }
}

ALWAYS_INLINE
static inline void sigmoid(float* inputs, float* results, int input_size) {
    sigmoid_loop:
    for (int i = 0; i < input_size; i++) {
        results[i] = 1.0 / (1.0 + exp(-inputs[i]));
    }
}

ALWAYS_INLINE
static inline void tanh_act(float* inputs, float* results, int input_size) {
    int i;
    tanh_act_loop1:
    for (i = 0; i < input_size; i++) {
        results[i] = 2 * inputs[i];
    }

    sigmoid(results, results, input_size);

    tanh_act_loop2:
    for (i = 0; i < input_size; i++) {
        results[i] = 2 * results[i] - 1;
    }
}

ALWAYS_INLINE
static inline void hard_tanh_act(
        float* inputs, float* results, int input_size, float min, float max) {
    hard_tanh_loop:
    for (int i = 0; i < input_size; i++) {
        float value = inputs[i];
        results[i] = (value < min) ? min : (value > max) ? max : value;
    }
}

ALWAYS_INLINE
static inline void activation_fun(float* inputs,
                                  float* results,
                                  int inputs_size,
                                  activation_type function,
                                  activation_param_t params) {
    if (function == RELU) {
        relu(inputs, results, inputs_size);
    } else if (function == LRELU) {
        lrelu(inputs, results, inputs_size, params.slope);
    } else if (function == ELU) {
        elu(inputs, results, inputs_size, params.alpha);
    } else if (function == SELU) {
        selu(inputs, results, inputs_size, params.alpha, params.lambda);
    } else if (function == TANH) {
        tanh_act(inputs, results, inputs_size);
    } else if (function == HARD_TANH) {
        hard_tanh_act(inputs, results, inputs_size, params.min, params.max);
    } else if (function == SIGMOID) {
        sigmoid(inputs, results, inputs_size);
    } else if (function == SOFTMAX) {
        assert(false && "Softmax not added yet!");
    }
}

/**
 * Top level entry point for all Reference activation functions.
 */
void ref_activation_fun_nc(float* inputs,
                           float* results,
                           int inputs_size,
                           activation_type function,
                           activation_param_t params);

#ifdef __cplusplus
}  // extern "C"
#endif

/**
 * @}
 */
#endif
