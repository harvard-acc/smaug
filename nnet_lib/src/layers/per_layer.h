#ifndef _PER_LAYER_H_
#define _PER_LAYER_H_

void nnet_fwd_per_layer(float* hid,
                        float* weights,
                        layer_t* layers,
                        int num_layers,
                        float* hid_temp,
                        float* sigmoid_table);
#endif
