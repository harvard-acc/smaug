#ifndef _ACTIVATION_FUNCTIONS_H_
#define _ACTIVATION_FUNCTIONS_H_

#include "nnet_fwd.h"

void activation_fun(float* hid,
                    int size,
                    activation_type function,
                    float* sigmoid_table);
void relu(float* a, int num_units);
void sigmoid_inplace(float* a, int num_units, float* sigmoid_table);
float sigmoid(float a);
void sigmoidn(float* a, int num_units);
void sigmoid_lookup(float* a, int num_units, float* sigmoid_table);
float* softmax(float* a, int num_test_cases, int num_classes);

#endif
