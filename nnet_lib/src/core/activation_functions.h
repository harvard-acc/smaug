#ifndef _ACTIVATION_FUNCTIONS_H_
#define _ACTIVATION_FUNCTIONS_H_

void RELU(float* a, int num_units);
float sigmoid(float a);
void sigmoidn(float* a, int num_units);
void sigmoid_lookup(float* a, int num_units, float* sigmoid_table);
float* softmax(float* a, int num_test_cases, int num_classes);

#endif
