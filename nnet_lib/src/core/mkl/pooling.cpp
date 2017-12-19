#include "mkldnn.hpp"

#include "core/mkl/pooling.h"

namespace nnet_mkl {

using namespace mkldnn;

void max_pooling_3d(float* inputs,
                    layer_t* curr_layer,
                    float* results,
                    device_t* device) {
    std::vector<primitive> network;
    nnet_mkl::MklSession* session =
            reinterpret_cast<nnet_mkl::MklSession*>(device->session);

    MaxPoolingOp<dtype> pooling_op(
            inputs, results, curr_layer, NUM_TEST_CASES, session->cpu);
    pooling_op.run();
}

} // namespace nnet_mkl
