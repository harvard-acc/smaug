#ifndef _UTILITY_H_
#define _UTILITY_H_

#include <stdbool.h>

#include "nnet_fwd.h"

float* grab_matrix(float* w, int n, int* n_rows, int* n_columns);
void grab_matrix_dma(float* weights, int layer, layer_t* layers);
void grab_input_activations_dma(float* activations, int layer, layer_t* layers);
void store_output_activations_dma(float* activations, int layer, layer_t* layers);
float randfloat();
void clear_matrix(float* input, int size);
void copy_matrix(float* input, float* output, int size);
int arg_max(float* input, int size, int increment);
int arg_min(float* input, int size, int increment);
int get_weights_offset_layer(layer_t* layers, int l);
void get_weights_dims_layer(layer_t* layers,
                            int l,
                            int* num_rows,
                            int* num_cols,
                            int* num_height,
                            int* num_depth);
int get_num_weights_layer(layer_t* layers, int l);
int get_total_num_weights(layer_t* layers, int num_layers);
int get_input_activations_size(layer_t* layers, int num_layers);
int get_output_activations_size(layer_t* layers, int num_layers);
bool is_dummy_layer(layer_t* layers, int l);
size_t next_multiple(size_t request, size_t align);

void print_debug(float* array,
                 int rows_to_print,
                 int cols_to_print,
                 int num_columns);
void print_debug4d(float* array, int rows, int cols, int height);
void print_data_and_weights(float* data, float* weights, layer_t first_layer);

#ifdef BITWIDTH_REDUCTION
// Don't add this function unless we want to model bit width quantization
// effects. In particular, do not enable this if we are building a trace.  We
// don't want to add in the cost of dynamically doing this operation - we
// assume the data is already in the specified reduced precision and all
// functional units are designed to handle that bitwidth. So just make this
// function go away.
float conv_float2fixed(float input);
#else
#define conv_float2fixed(X) (X)
#endif

#endif
