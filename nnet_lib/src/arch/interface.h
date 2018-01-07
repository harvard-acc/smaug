#ifndef _LAYERS_INTERFACE_H_
#define _LAYERS_INTERFACE_H_

#include "nnet_fwd.h"

// This header defines the interface that every neural network architecture
// must implement.
//
// All architectures are expected to use the common run_layers dispatch
// function, which is responsible for calling the *_layer functions defined
// below.
//
// Furthermore, each architecture implementation must protect the definition
// file with an #if guard, based on the value of ARCHITECTURE, to ensure that
// the program does not contain multiple definitions of a function.
//
// Finally, to simplify matters for Aladdin, naming of function arguments is
// consistent throughout (the "activations" array is never called "input" or
// some other name, and so on).

result_buf inner_product_layer(float* activations,
                               float* weights,
                               layer_t* layers,
                               int lnum,
                               float* result,
                               device_t* device,
                               sampling_param_t* sampling_param);

result_buf standard_convolution_layer(float* activations,
                                      float* weights,
                                      layer_t* layers,
                                      int lnum,
                                      float* result,
                                      device_t* device,
                                      sampling_param_t* sampling_param);

result_buf depthwise_convolution_layer(float* activations,
                                       float* weights,
                                       layer_t* layers,
                                       int lnum,
                                       float* result,
                                       device_t* device,
                                       sampling_param_t* sampling_param);

result_buf pointwise_convolution_layer(float* activations,
                                       float* weights,
                                       layer_t* layers,
                                       int lnum,
                                       float* result,
                                       device_t* device,
                                       sampling_param_t* sampling_param);

result_buf pooling_layer(float* activations,
                         layer_t* layers,
                         int lnum,
                         float* result,
                         device_t* device,
                         sampling_param_t* sampling_param);

result_buf batch_norm_layer(float* activations,
                            float* weights,
                            layer_t* layers,
                            int lnum,
                            float* result,
                            device_t* device,
                            sampling_param_t* sampling_param);

result_buf flatten_input(float* activations,
                         layer_t* layers,
                         int lnum,
                         float* result);

/*
result_buf activation_sublayer(float* activations,
                               layer_t* layers,
                               int lnum);
                               */

// Does the forward predictive pass of a neural net.
//
// A float array of class predictions in row major format of size
// num_test_cases*num_labels will eventually be stored in either @hid or
// @hid_temp.
//
// A bool indicating where the final result is stored into the layers
// structure. If it is in @hid, then false, if in @hid_temp, true.
void nnet_fwd(farray_t activations,
              farray_t weights,
              farray_t result,
              network_t network,
              device_t* device,
              sampling_param_t* sampling_param);

#endif
