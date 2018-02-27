#ifndef _LOOKUP_TABLES_OPS_H_
#define _LOOKUP_TABLES_OPS_H_

#include "config.h"
#include "utility/utility.h"

ALWAYS_INLINE
static inline float exp_lut_fxp(float a) {
    float result;
    if (a > EXP_MAX) {
        result = exp_table[EXP_TABLE_SIZE - 1];
    } else if (a < EXP_MIN) {
        result = 0;
    } else {
        float temp = conv_float2fixed(((a - EXP_MIN) * (1.0 / EXP_RANGE)) *
                                      (EXP_TABLE_SIZE - 1.0));
        int ind = (int)temp;  // Ideally a proper rounding.
        result = conv_float2fixed(exp_table[ind]);
    }
    return result;
}

// The logistic activation function, implemented with a lookup table
// and linear interpolation
// ** this function is in-place (modifies a) **
ALWAYS_INLINE
static inline float sigmoid_lookup_centered_op_fxp(float a) {
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
static inline float sigmoid_lookup_noncentered_op_fxp(float a) {
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

#endif
