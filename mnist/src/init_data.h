#ifndef _INIT_DATA_H_
#define _INIT_DATA_H_

#include <stdbool.h>
#include "nnet_fwd.h"

void init_weights(float* weights,
                  layer_t* layers,
                  int num_layers,
                  bool random,
                  bool transpose);
void init_data(float* data,
               size_t num_test_cases,
               size_t input_dim,
               bool random);
void init_labels(int* labels, size_t label_size, bool random);
void init_kernels(float* kernels, size_t kernel_size);

#endif
