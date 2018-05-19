#ifndef _ACTIVATION_FUNCTIONS_H_
#define _ACTIVATION_FUNCTIONS_H_

#include "nnet_fwd.h"

void activation_fun(float* hid,
                    int batch_size,
                    int input_size,
                    int input_pad,
                    activation_type function);
void activation_fun_fxp(float* hid,
                        int batch_size,
                        int input_size,
                        int input_pad,
                        activation_type function);
void relu(float* a, int num_units);
void lrelu(float* a, int num_units, float alpha);
void elu(float* a, int num_units, float alpha, float* results);
void selu(float* a, int num_units);
void hard_tanh(float* a, int num_units, float min, float max, float* results);
void tanh_act(float* a, int num_units, float* results);
void sigmoid_inplace(float* a, int num_units);
float sigmoid_fxp(float a);
void sigmoidn(float* a, int num_units);
void sigmoid_lookup(float* a, int num_units);
void softmax(float* a, int num_test_cases, int softmax_size, int input_pad);

void elu_lut_fxp(float* a, int num_units, float alpha, float* results);
void sigmoid_lookup_centered(float* a, int num_units, float* results);
void sigmoid_lookup_noncentered(float* a, int num_units, float* results);

#endif
