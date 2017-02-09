#ifndef _CONVOLUTION_H_
#define _CONVOLUTION_H_

#include "nnet_fwd.h"

void convolution2d_zeropad(float* input,
                           float* kernels,
                           layer_t curr_layer,
                           float* result);
void convolution2d_no_padding(float* a,
                              float* kernels,
                              layer_t curr_layer,
                              float* result);
void convolution2d_kernel_no_padding(float* a,
                                     float* kernels,
                                     int img,
                                     int kern,
                                     layer_t curr_layer,
                                     float* result);
#endif
