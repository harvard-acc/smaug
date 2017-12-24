#ifndef _EIGEN_INIT_DATA_H_
#define _EIGEN_INIT_DATA_H_

#include "core/nnet_fwd_defs.h"

namespace nnet_eigen {

void init_fc_weights(float* weights,
                     int w_height,
                     int w_rows,
                     int w_cols,
                     data_init_mode mode);

void init_conv_weights(float* weights,
                       int w_depth,
                       int w_height,
                       int w_rows,
                       int w_cols,
                       data_init_mode mode);

void init_bn_weights(float* weights,
                     int w_height,
                     int w_rows,
                     int w_cols,
                     data_init_mode mode);

void init_weights(float* weights,
                  layer_t* layers,
                  int num_layers,
                  data_init_mode mode);

void init_data(float* data,
               network_t* network,
               int num_test_cases,
               data_init_mode mode);

void init_labels(int* labels, size_t label_size, data_init_mode mode);

}  // namespace nnet_eigen

#endif
