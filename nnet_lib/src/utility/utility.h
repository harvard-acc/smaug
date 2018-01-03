#ifndef _UTILITY_H_
#define _UTILITY_H_

#include <stdbool.h>

#include "nnet_fwd.h"

float* grab_matrix(float* w, int n, int* n_rows, int* n_columns);
size_t get_weights_loc_for_layer(layer_t* layers, int layer);

#if defined(DMA_INTERFACE_V2)
#define INPUT_BYTES(layers, lnum)                                              \
    get_input_activations_size(layers, lnum) * sizeof(float)
#define OUTPUT_BYTES(layers, lnum)                                             \
    get_output_activations_size(layers, lnum) * sizeof(float)
#define WEIGHT_BYTES(layers, lnum)                                             \
    get_num_weights_layer(layers, lnum) * sizeof(float)

int get_input_activations_size(layer_t* layers, int num_layers);
int get_output_activations_size(layer_t* layers, int num_layers);
void grab_weights_dma(float* weights, int layer, layer_t* layers);
size_t grab_input_activations_dma(float* activations, int layer, layer_t* layers);
size_t grab_output_activations_dma(float* activations, int layer, layer_t* layers);
size_t store_output_activations_dma(float* activations, int layer, layer_t* layers);

#elif defined(DMA_INTERFACE_V3)
#define INPUT_BYTES(layers, lnum)                                              \
    get_input_activations_size(&layers[lnum]) * sizeof(float)
#define OUTPUT_BYTES(layers, lnum)                                             \
    get_output_activations_size(&layers[lnum]) * sizeof(float)
#define WEIGHT_BYTES(layers, lnum)                                             \
    get_num_weights_layer(layers, lnum) * sizeof(float)

int get_input_activations_size(layer_t* layer);
int get_output_activations_size(layer_t* layer);
void grab_weights_dma(float* host_weights,
                      float* accel_weights,
                      int layer,
                      layer_t* layers);
size_t grab_output_activations_dma(float* host_activations,
                                   float* accel_activations,
                                   layer_t* layer);
size_t grab_input_activations_dma(float* host_activations,
                                  float* accel_activations,
                                  layer_t* layer);
size_t store_output_activations_dma(float* host_activations,
                                    float* accel_activations,
                                    layer_t* layer);
#endif

float randfloat();
void clear_matrix(float* input, int size);
void copy_matrix(float* input, float* output, int size);
int arg_max(float* input, int size, int increment);
int arg_min(float* input, int size, int increment);
int calc_padding(int value, unsigned alignment);
int get_weights_offset_layer(layer_t* layers, int l);
void get_weights_dims_layer(layer_t* layers,
                            int l,
                            int* num_rows,
                            int* num_cols,
                            int* num_height,
                            int* num_depth,
                            int* num_pad);
void get_unpadded_inputs_dims_layer(layer_t* layers,
                                    int l,
                                    int* num_rows,
                                    int* num_cols,
                                    int* num_height,
                                    int* pad_amt);
int get_num_weights_layer(layer_t* layers, int l);
int get_total_num_weights(layer_t* layers, int num_layers);
bool is_dummy_layer(layer_t* layers, int l);
size_t next_multiple(size_t request, size_t align);
size_t get_dims_size(dims_t* dims);

float compute_errors(float* network_pred,
                     int* correct_labels,
                     int batch_size,
                     int num_classes);
void write_output_labels(const char* fname,
                         float* network_pred,
                         int batch_size,
                         int num_classes,
                         int pred_pad);

void print_debug(float* array,
                 int rows_to_print,
                 int cols_to_print,
                 int num_columns);
void print_debug4d(float* array, int rows, int cols, int height);
void print_data_and_weights(float* data, float* weights, layer_t first_layer);

void* malloc_aligned(size_t size);

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
