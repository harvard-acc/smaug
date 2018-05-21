#ifndef _OPERATORS_REF_REF_COMMON_H_
#define _OPERATORS_REF_REF_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif

void ref_tanh_f32(float* inputs, float* results, int input_size);
void ref_sigmoid_f32(float* inputs, float* results, int input_size);
void ref_hard_tanh_f32(
        float* inputs, float* results, int input_size, float min, float max);
void ref_elu_f32(float* inputs, float* results, int input_size, float alpha);
void ref_selu_f32(float* inputs,
                  float* results,
                  int input_size,
                  float alpha,
                  float lambda);

#ifdef __cplusplus
}
#endif

#endif
