#ifndef _FLATTEN_H_
#define _FLATTEN_H_

#include "nnet_fwd.h"

int im2row_size(layer_t* layers, int lnum);
data_list* im2row(data_list* input, layer_t* layers, int lnum, data_list* result);

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

size_t compute_blocked_nhwc_size(dims_t* input_dims,
                                int block_size,
                                int data_alignment);

int convert_nchw_to_blocked_nhwc(float* input,
                                  int num_inputs,
                                  int block_size,
                                  dims_t input_dims,
                                  unsigned data_alignment,
                                  float** result);

int convert_blocked_nhwc_to_nchw(float* input,
                                  int num_inputs,
                                  int block_size,
                                  dims_t input_dims,
                                  unsigned data_alignment,
                                  float** result);
#endif
