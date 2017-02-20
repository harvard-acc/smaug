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

// Get the dimensions of this layer's weights.
//
// Store them into @num_rows and @num_cols.
void get_weights_dims_layer(layer_t* layers,
                           int l,
                           int* num_rows,
                           int* num_cols,
                           int* num_height,
                           int* num_depth) {

    if (layers[l].type == FC) {
        *num_rows = layers[l].input_rows;
        *num_cols = layers[l].input_cols;
        *num_height = 1;
        *num_depth = 1;
    } else if (layers[l].type == CONV) {
        *num_rows = layers[l].field_size;
        *num_cols = layers[l].field_size;
        *num_height = layers[l].input_height;  // 3rd dim of input.
        *num_depth = layers[l].output_height;  // # of this layer's kernels.
    } else {
        *num_rows = 0;
        *num_cols = 0;
        *num_height = 0;
        *num_depth = 0;
    }
}

// Get the total number of weights for layer @l in the network.
int get_num_weights_layer(layer_t* layers, int l) {
    if (layers[l].type == FC)
        return layers[l].input_rows * layers[l].input_cols;
    else if (layers[l].type == CONV)
        return layers[l].output_height * layers[l].input_height *
               layers[l].field_size * layers[l].field_size;
    else
        return 0;
}

// Get the total number of weights for the entire network.
int get_total_num_weights(layer_t* layers, int num_layers) {
    int l;
    int w_size = 0;
    for (l = 0; l < num_layers; l++) {
        w_size += get_num_weights_layer(layers, l);
    }
    return w_size;
}

size_t next_multiple(size_t request, size_t align) {
  size_t n = request/align;
  if (n == 0)
    return align;  // Return at least this many bytes.
  size_t remainder = request % align;
  if (remainder)
      return (n+1)*align;
  return request;
}
