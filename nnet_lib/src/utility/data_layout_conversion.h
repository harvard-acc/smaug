#ifndef _FLATTEN_H_
#define _FLATTEN_H_

#include "nnet_fwd.h"

//=----------------------- NCHW to rows -----------------------=//

int im2row_size(layer_t* layers, int lnum);
data_list* im2row(data_list* input, layer_t* layers, int lnum, data_list* result);

//=---------------------- Dimension conversions ----------------------=//

dims_t nchw_to_nhwc_dims(dims_t* input_dims, unsigned data_alignment);
dims_t nhwc_to_nchw_dims(dims_t* input_dims, unsigned data_alignment);

//=----------- Channel (NCHW) first to channel last (NHWC) ------------=//

dims_t convert_nchw_to_nhwc(data_list* input,
                            int data_index,
                            int num_inputs,
                            dims_t input_dims,
                            unsigned data_alignment,
                            data_list* result);
dims_t convert_nchw_to_nhwc_fp32(float* input,
                                 int num_inputs,
                                 dims_t input_dims,
                                 unsigned data_alignment,
                                 float** result);
dims_t convert_nchw_to_nhwc_fp16(packed_fp16* input,
                                 int num_inputs,
                                 dims_t input_dims,
                                 unsigned data_alignment,
                                 packed_fp16** result);
dims_t convert_nchw_to_nhwc_farray(farray_t* input,
                                   int num_inputs,
                                   dims_t input_dims,
                                   unsigned data_alignment,
                                   farray_t** result);
dims_t convert_nchw_to_nhwc_fp16array(fp16array_t* input,
                                      int num_inputs,
                                      dims_t input_dims,
                                      unsigned data_alignment,
                                      fp16array_t** result);

//=----------- Channel last (NHWC) to channel first (NCHW) ------------=//

dims_t convert_nhwc_to_nchw(data_list* input,
                            int data_index,
                            int num_inputs,
                            dims_t input_dims,
                            unsigned data_alignment,
                            data_list* result);
dims_t convert_nhwc_to_nchw_fp32(float* input,
                                 int num_inputs,
                                 dims_t input_dims,
                                 unsigned data_alignment,
                                 float** result);
dims_t convert_nhwc_to_nchw_fp16(packed_fp16* input,
                                 int num_inputs,
                                 dims_t input_dims,
                                 unsigned data_alignment,
                                 packed_fp16** result);
dims_t convert_nhwc_to_nchw_farray(farray_t* input,
                                   int num_inputs,
                                   dims_t input_dims,
                                   unsigned data_alignment,
                                   farray_t** result);
dims_t convert_nhwc_to_nchw_fp16array(fp16array_t* input,
                                      int num_inputs,
                                      dims_t input_dims,
                                      unsigned data_alignment,
                                      fp16array_t** result);

//=--------- Blocked channel first (NCHW) to channel last (NHWC) ---------=//

size_t compute_blocked_nhwc_size(dims_t* input_dims,
                                 int block_size,
                                 int data_alignment);
int convert_nchw_to_blocked_nhwc(data_list* input,
                                 int data_index,
                                 int num_inputs,
                                 int block_size,
                                 dims_t input_dims,
                                 unsigned data_alignment,
                                 data_list* result);
int convert_nchw_to_blocked_nhwc_fp16(fp16array_t* input,
                                      int num_inputs,
                                      int block_size,
                                      dims_t input_dims,
                                      unsigned data_alignment,
                                      fp16array_t** result);
int convert_nchw_to_blocked_nhwc_fp32(farray_t* input,
                                      int num_inputs,
                                      int block_size,
                                      dims_t input_dims,
                                      unsigned data_alignment,
                                      farray_t** result);

//=--------- Blocked channel last (NHWC) to channel first (NCHW) ---------=//

int convert_blocked_nhwc_to_nchw(data_list* input,
                                 int data_index,
                                 int num_inputs,
                                 int block_size,
                                 dims_t input_dims,
                                 unsigned data_alignment,
                                 data_list* result);
int convert_blocked_nhwc_to_nchw_fp16(fp16array_t* input,
                                      int num_inputs,
                                      int block_size,
                                      dims_t input_dims,
                                      unsigned data_alignment,
                                      fp16array_t** result);
int convert_blocked_nhwc_to_nchw_fp32(farray_t* input,
                                      int num_inputs,
                                      int block_size,
                                      dims_t input_dims,
                                      unsigned data_alignment,
                                      farray_t** result);
#endif
