#include "utility.h"

float randfloat() { return rand() / ((float)(RAND_MAX)); }

float conv_float2fixed(float input) {
    // return input;
    int sign = 1;
    if (input < 0) {
        sign = -1;
    }
    long long int long_1 = 1;

    return sign *
           ((float)((long long int)(fabs(input) *
                                    (long_1 << NUM_OF_FRAC_BITS)) &
                    ((long_1 << (NUM_OF_INT_BITS + NUM_OF_FRAC_BITS)) - 1))) /
           (long_1 << NUM_OF_FRAC_BITS);
}

void clear_matrix(float* input, int size) {
    int i;
    for (i = 0; i < size; i++)
        input[i] = 0.0;
}

void copy_matrix(float* input, float* output, int size) {
    int i;
    for (i = 0; i < size; i++)
        output[i] = input[i];
}

int arg_max(float* input, int size, int increment) {
    int i;
    int j = 0;
    int max_ind = 0;
    float max_val = input[0];
    for (i = 1; i < size; i++) {
        j += increment;
        if (input[j] > max_val) {
            max_ind = i;
            max_val = input[j];
        }
    }
    return max_ind;
}

int arg_min(float* input, int size, int increment) {
    int i;
    int j = 0;
    int min_ind = 0;
    float min_val = input[0];
    for (i = 1; i < size; i++) {
        j += increment;
        if (input[j] < min_val) {
            min_ind = i;
            min_val = input[j];
        }
    }
    return min_ind;
}

// Softmax function on matrix a
// a is num_test_cases by num_classes
// the softmax function exponentiates each element and then normalizes each row
// to sum to 1
// ** this function is in-place (modifies a) **
float* softmax(float* a, int num_test_cases, int num_classes) {

    int i, j;
    float numerator, normaliz;
    for (i = 0; i < num_test_cases; i++) {
        // compute the normalization factor
        normaliz = 0.0;
        for (j = 0; j < num_classes; j++) {
            numerator = sigmoid(a[sub2ind(i, j, num_classes)]);
            // replace a[i,j] with exp(a[i,j])
            a[sub2ind(i, j, num_classes)] = numerator;
            normaliz += numerator;
        }
        for (j = 0; j < num_classes; j++) {
            a[sub2ind(i, j, num_classes)] /= normaliz;
        }
    }
    return a;
}
