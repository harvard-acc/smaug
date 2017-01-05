#ifndef _UTILITY_H_
#define _UTILITY_H_

float randfloat();
float conv_float2fixed(float input);
void clear_matrix(float* input, int size);
void copy_matrix(float* input, float* output, int size);
int arg_max(float* input, int size, int increment);
int arg_min(float* input, int size, int increment);
float* softmax(float* a, int num_test_cases, int num_classes);

#endif
