#ifndef _SMIV_IMPL_H_
#define _SMIV_IMPL_H_

#include "core/smiv/params.h"

void convolution3d_smiv_1kernel_noreduce_fxp(float* a,
                                             float* kernels,
                                             layer_t curr_layer,
                                             int start_chan,
                                             float* result);

void convolution3d_smiv_1kernel_noreduce_simd_fxp(float* a,
                                                  float* kernels,
                                                  layer_t curr_layer,
                                                  int start_chan,
                                                  float* result);

void reduction_smiv_fxp(float* a, layer_t curr_layer, float* result);

void reduction_smiv_vec_fxp(float* a, layer_t curr_layer, float* result);

void matrix_multiply_with_bias_smiv_batch_fxp(float* a,
                                              float* b,
                                              int a_height,
                                              int b_height,
                                              int b_width,
                                              int a_pad,
                                              activation_type act_func,
                                              bool do_bias,
                                              float* result);

void matrix_multiply_with_bias_smiv_nobatch_fxp(float* a,
                                                float* b,
                                                int a_height,
                                                int b_height,
                                                int b_width,
                                                int a_pad,
                                                activation_type act_func,
                                                bool do_bias,
                                                float* result);

void matrix_multiply_with_bias_smiv_nobatch_vec_fxp(float* a,
                                                    float* b,
                                                    int a_height,
                                                    int b_height,
                                                    int b_width,
                                                    int a_pad,
                                                    activation_type act_func,
                                                    bool do_bias,
                                                    float* result);

// Compute a RELU with vector literals.
//
// With vector literals, the comparison returns a vector of signed ints, each
// of which are either all zero or all one.  An easy way to implement RELU is
// just to bitwise AND the comparison result with the partial sums, which
// requires some casting to and from integer and fp vector types.
//
// If we made this an actual function, we would need to return a vector literal
// by value (we can't use pointers with Aladdin since this is not a pre-declared
// array), and returning a vector by value requires support for AVX on the
// host, which we're not guaranteed to have. Instead, make this a macro so we can
// get code reuse.
#define RELU_VEC_SMIV(activations)                                             \
    do {                                                                       \
        v8fp_t zero = (v8fp_t){ 0 };                                           \
        v8sfx_t mask = (activations > zero);                                   \
        activations = ((v8fp_t)((v8sfx_t)activations & mask));                 \
    } while (0)

#endif
