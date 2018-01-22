#ifndef _ACTIVATION_FUNCTIONS_SIMD_H_
#define _ACTIVATION_FUNCTIONS_SIMD_H_

#include "core/nnet_fwd_defs.h"
#include "core/smiv/params.h"

v8fp_t activation_fun_simd_fxp(v8fp_t activations, activation_type function);
v8fp_t activation_fun_simd(v8fp_t activations, activation_type function);
v8fp_t relu_simd(v8fp_t a);
v8fp_t lrelu_simd(v8fp_t a);
v8fp_t elu_simd(v8fp_t a, float alpha);
v8fp_t selu_simd(v8fp_t a);
v8fp_t tanh_act_simd(v8fp_t a);
v8fp_t sigmoid_inplace_simd(v8fp_t a);
v8fp_t sigmoidn_simd(v8fp_t a);
v8fp_t sigmoid_lookup_centered_simd(v8fp_t a);
v8fp_t sigmoid_lookup_noncentered_simd(v8fp_t a);

#endif
