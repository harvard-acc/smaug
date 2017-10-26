#ifndef _ZEROPAD_H_
#define _ZEROPAD_H_

#include "nnet_fwd.h"

void copy_zeropad(float* a, layer_t* layers, int lnum, float* result);
void copy_zeropad_image3d(float* a,
                          int pad,
                          int img,
                          int fmap,
                          int a_width,
                          int a_height,
                          int a_data_pad,
                          float* result,
                          int r_width,
                          int r_height,
                          int r_data_pad);

#endif
