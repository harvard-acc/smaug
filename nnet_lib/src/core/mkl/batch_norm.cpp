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
    std::vector<primitive> network;
    nnet_mkl::MklSession* session =
            reinterpret_cast<nnet_mkl::MklSession*>(device->session);

    BatchNormOp<dtype> batch_norm_op(
            inputs, weights, results, curr_layer, batch_size, session->cpu);
    batch_norm_op.run();
}

}  // namespace nnet_mkl
