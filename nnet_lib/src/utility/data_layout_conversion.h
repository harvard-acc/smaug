#ifndef _FLATTEN_H_
#define _FLATTEN_H_

#include "nnet_fwd.h"

result_buf im2row(float* input, layer_t* layers, int lnum, float* result);

dims_t nchw_to_nhwc_dims(dims_t* input_dims, unsigned data_alignment);
dims_t nhwc_to_nchw_dims(dims_t* input_dims, unsigned data_alignment);

dims_t convert_nchw_to_nhwc(float* input,
                            int num_inputs,
                            dims_t input_dims,
                            unsigned data_alignment,
                            float** result);

dims_t convert_nhwc_to_nchw(float* input,
                            int num_inputs,
                            dims_t input_dims,
                            unsigned data_alignment,
                            float** result);
#endif
