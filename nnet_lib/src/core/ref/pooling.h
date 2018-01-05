#ifndef _POOLING_H_
#define _POOLING_H_

#include "nnet_fwd.h"

void max_pooling(float* input, float* result, layer_t curr_layer);
void max_pooling_image3d(float* input, int ni, float* result, layer_t curr_layer);
void avg_pooling(float* input, float* result, layer_t curr_layer);
void avg_pooling_image3d(float* input, int ni, float* result, layer_t curr_layer);

#endif
