#ifndef _FLATTEN_H_
#define _FLATTEN_H_

#include "nnet_fwd.h"

result_buf flatten_input_rowmajor(float* input,
                                  layer_t* layers,
                                  int lnum,
                                  float* result);

void im2row(float* input,
            int input_rows,
            int input_cols,
            int input_height,
            int input_pad,
            int output_pad,
            float* result);

#endif
