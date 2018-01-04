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
    auto session = get_session(device);
    if (session->empty()) {
        session->add_op(new Convolution3dOp<dtype>(inputs,
                                                   weights,
                                                   results,
                                                   curr_layer,
                                                   NUM_TEST_CASES,
                                                   session->cpu()));
    } else {
        session->add_op(new Convolution3dOp<dtype>(session->last_op(),
                                                   weights,
                                                   results,
                                                   curr_layer,
                                                   NUM_TEST_CASES,
                                                   session->cpu()));
    }
}

void depthwise_convolution3d(float* inputs,
                             float* weights,
                             layer_t* curr_layer,
                             float* results,
                             device_t* device) {
    auto session = get_session(device);
    if (session->empty()) {
        session->add_op(new DepthwiseConvolution3dOp<dtype>(inputs,
                                                            weights,
                                                            results,
                                                            curr_layer,
                                                            NUM_TEST_CASES,
                                                            session->cpu()));
    } else {
        session->add_op(new DepthwiseConvolution3dOp<dtype>(session->last_op(),
                                                            weights,
                                                            results,
                                                            curr_layer,
                                                            NUM_TEST_CASES,
                                                            session->cpu()));
    }
}

void pointwise_convolution3d(float* inputs,
                             float* weights,
                             layer_t* curr_layer,
                             float* results,
                             device_t* device) {
    auto session = get_session(device);
    if (session->empty()) {
        session->add_op(new PointwiseConvolution3dOp<dtype>(inputs,
                                                            weights,
                                                            results,
                                                            curr_layer,
                                                            NUM_TEST_CASES,
                                                            session->cpu()));
    } else {
        session->add_op(new PointwiseConvolution3dOp<dtype>(session->last_op(),
                                                            weights,
                                                            results,
                                                            curr_layer,
                                                            NUM_TEST_CASES,
                                                            session->cpu()));
    }
}

}  // namespace nnet_mkl
