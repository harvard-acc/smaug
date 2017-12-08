#ifndef _INIT_DATA_H_
#define _INIT_DATA_H_

#include <stdbool.h>
#include "nnet_fwd.h"

void init_fc_weights(float* weights,
                     int w_height,
                     int w_rows,
                     int w_cols,
                     int w_pad,
                     data_init_mode mode,
                     bool transpose);

void init_conv_weights(float* weights,
                       int w_depth,
                       int w_height,
                       int w_rows,
                       int w_cols,
                       int w_pad,
                       data_init_mode mode,
                       bool transpose);

void init_bn_weights(float* weights,
                     int w_height,
                     int w_rows,
                     int w_cols,
                     int w_pad,
                     data_init_mode mode,
                     bool transpose);

void init_weights(float* weights,
                  layer_t* layers,
                  int num_layers,
                  data_init_mode mode,
                  bool transpose);
void init_data(float* data,
               network_t* network,
               size_t num_test_cases,
               data_init_mode mode);
void init_labels(int* labels, size_t label_size, data_init_mode mode);
void init_kernels(float* kernels, size_t kernel_size);

#endif
