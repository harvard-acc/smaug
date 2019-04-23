#ifndef _OPERATORS_REF_REF_COMMON_H_
#define _OPERATORS_REF_REF_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif

void ref_tanh(float* inputs, float* results, int input_size);
void ref_sigmoid(float* inputs, float* results, int input_size);
void ref_hard_tanh(
        float* inputs, float* results, int input_size, float min, float max);
void ref_elu(float* inputs, float* results, int input_size, float alpha);
void ref_selu(float* inputs,
              float* results,
              int input_size,
              float alpha,
              float lambda);

#ifdef __cplusplus
}
#endif

#endif
