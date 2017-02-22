#ifndef _COMPOSABLE_H_
#define _COMPOSABLE_H_

#include "nnet_fwd.h"

void inner_product_layer_hw(float* activations,
                            float* weights,
                            layer_t* layers,
                            int lnum,
                            float* result);

result_buf inner_product_layer_sw(float* activations,
                                  float* weights,
                                  layer_t* layers,
                                  int lnum,
                                  float* result);

void convolution_layer_hw(float* input,
                          float* kernels,
                          layer_t* layers,
                          int lnum,
                          float* result);

result_buf convolution_layer_sw(float* input,
                                float* kernels,
                                layer_t* layers,
                                int lnum,
                                float* result);

void max_pooling_layer_hw(float* input,
                          float* result,
                          layer_t* layers,
                          int lnum);

result_buf max_pooling_layer_sw(float* input,
                                layer_t* layers,
                                int lnum,
                                float* result);

result_buf run_layer(float* activations,
                     float* weights,
                     layer_t* layers,
                     int layer_num,
                     float* result_temp,
                     float* sigmoid_table,
                     bool do_activation_func);

void nnet_fwd_composable(float* hid,
                         float* weights,
                         layer_t* layers,
                         int num_layers,
                         float* hid_temp,
                         float* sigmoid_table);

#endif
