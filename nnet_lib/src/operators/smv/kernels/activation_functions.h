#ifndef _SMV_ACTIVATION_FUNCTIONS_H_
#define _SMV_ACTIVATION_FUNCTIONS_H_

#include "core/nnet_fwd_defs.h"

void relu_simd128(float* inputs, size_t size);
void lrelu_simd128(float* inputs, size_t size, float alpha);
void tanh_act_simd128(float* inputs, int size, float* results);
void hard_tanh_simd128(
        float* inputs, int size, float min, float max, float* results);
void activation_fun_simd128(packed_fp16* activations,
                            int batch_size,
                            layer_t* layer,
                            dims_t* input_dims,
                            activation_type function,
                            packed_fp16* results);

#endif
