#include <memory>

#include "mkldnn.hpp"

#include "core/mkl/convolution.h"
#include "core/nnet_fwd_defs.h"
#include "arch/nnet_mkl.h"

namespace nnet_mkl {

using namespace mkldnn;

void convolution3d(float* inputs,
                   float* weights,
                   layer_t* curr_layer,
                   float* results,
                   device_t* device) {
    std::vector<primitive> network;
    nnet_mkl::MklSession* session =
            reinterpret_cast<nnet_mkl::MklSession*>(device->session);

    Convolution3dOp<dtype> conv_op(
            inputs, weights, results, curr_layer, NUM_TEST_CASES, session->cpu);
    conv_op.run();
}

}  // namespace nnet_mkl
