#ifndef _ACTIVATION_FUNCTIONS_H_
#define _ACTIVATION_FUNCTIONS_H_

#include "nnet_fwd.h"

void init_sigmoid_table(float** table_ptr);
void init_exp_table(float** table_ptr);
float exp_lut(float a);
float sigmoid_lookup_centered_op(float a);
float sigmoid_lookup_noncentered_op(float a);

void activation_fun(float* hid,
                    int batch_size,
                    int input_size,
                    activation_type function);
void relu(float* a, int num_units);
void lrelu(float* a, int num_units);
void elu(float* a, int num_units, float alpha);
void selu(float* a, int num_units);
void tanh_act(float* a, int num_units);
void sigmoid_inplace(float* a, int num_units);
float sigmoid(float a);
void sigmoidn(float* a, int num_units);
void sigmoid_lookup(float* a, int num_units);
void softmax(float* a, int num_test_cases, int softmax_size);

void sigmoid_lookup_centered(float* a, int num_units);
void sigmoid_lookup_noncentered(float* a, int num_units);

#endif
