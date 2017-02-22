#ifndef _PER_LAYER_H_
#define _PER_LAYER_H_

bool run_layer_m(float* activations,
               float* weights,
               layer_t curr_layer,
               float* result_temp,
               float* sigmoid_table,
               bool do_activation_func);

void nnet_fwd_monolithic(float* hid,
                         float* weights,
                         layer_t* layers,
                         int num_layers,
                         float* hid_temp,
                         float* sigmoid_table);
#endif
