#ifndef _UTILITY_H_
#define _UTILITY_H_

float randfloat();
void clear_matrix(float* input, int size);
void copy_matrix(float* input, float* output, int size);
int arg_max(float* input, int size, int increment);
int arg_min(float* input, int size, int increment);

#ifndef TRACE_MODE
// If we're building a dynamic trace, we don't want to add in the cost of doing
// this - we assume the data is already in the specified reduced precision and
// all functional units are designed to handle that bitwidth. So just make this
// function go away.
float conv_float2fixed(float input);
#else
#define conv_float2fixed(X) X
#endif

#endif
