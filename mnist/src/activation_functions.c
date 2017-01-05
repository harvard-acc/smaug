#include "activation_functions.h"

// The rectified linear activation function
// ** this function is in-place (modifies a) **
void RELU(float* a, int num_units) {
    int i;
    for (i = 0; i < num_units; i++) {
        if (a[i] < 0.0) {
            a[i] = 0.0;
        }
    }
}

// The logistic activation function
// ** this function is in-place (modifies a) **
void sigmoid(float* a, int num_units) {
    int i;
    float value;
    for (i = 0; i < num_units; i++) {
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
