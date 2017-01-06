#ifndef _INIT_DATA_H_
#define _INIT_DATA_H_

#include <stdbool.h>

void init_weights(float* weights, size_t w_size, bool random);
void init_data(float* data,
               size_t num_test_cases,
               size_t input_dim,
               bool random);
void init_labels(int* labels, size_t label_size, bool random);

#endif
