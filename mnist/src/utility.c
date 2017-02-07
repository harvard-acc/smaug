#include "nnet_fwd.h"
#include "utility.h"

float randfloat() { return rand() / ((float)(RAND_MAX)); }

#ifdef BITWIDTH_REDUCTION
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
#endif

void clear_matrix(float* input, int size) {
    int i;
clear_loop:    for (i = 0; i < size; i++)
        input[i] = 0.0;
}

void copy_matrix(float* input, float* output, int size) {
    int i;
copy_loop:    for (i = 0; i < size; i++)
        output[i] = input[i];
}

int arg_max(float* input, int size, int increment) {
    int i;
    int j = 0;
    int max_ind = 0;
    float max_val = input[0];
arg_max_loop:    for (i = 1; i < size; i++) {
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
arg_min_loop:    for (i = 1; i < size; i++) {
        j += increment;
        if (input[j] < min_val) {
            min_ind = i;
            min_val = input[j];
        }
    }
    return min_ind;
}

int get_num_weights_layer(layer_t* layers, int l) {
    if (layers[l].type == FC)
        return layers[l].input_rows * layers[l].input_cols;
    else if (layers[l].type == CONV)
        return layers[l].c_num_kernels * layers[l].c_kernel_size *
               layers[l].c_kernel_size;
    else
        return 0;
}

int get_total_num_weights(layer_t* layers, int num_layers) {
    int l;
    int w_size = 0;
    for (l = 0; l < num_layers; l++) {
        w_size += get_num_weights_layer(layers, l);
    }
    return w_size;
}

bool is_dummy_layer(layer_t* layers, int l) {
    switch (layers[l].type) {
        case FLATTEN:
        case INPUT:
            return true;
        default:
            return false;
    }
}
