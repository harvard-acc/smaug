#include <iostream>

#include "core/mkl/batch_norm.h"
#include "utility/utility.h"

namespace nnet_mkl {

using namespace mkldnn;

void batch_norm(float* inputs,
                float* weights,
                layer_t* curr_layer,
                int batch_size,
                float* results,
                device_t* device) {
    auto session = get_session(device);
    session->add_op(new BatchNormOp<dtype>(
            inputs, weights, results, curr_layer, batch_size, session->cpu()));
}

}  // namespace nnet_mkl
