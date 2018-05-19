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
    auto op = std::make_unique<Convolution3dOp<dtype>>(
            curr_layer, NUM_TEST_CASES, session->cpu());
    if (session->empty()) {
        op->init(inputs, weights, results);
    } else {
        op->init(*session->last_op(), weights, results);
    }
    session->push_back(std::move(op));
}

void depthwise_convolution3d(float* inputs,
                             float* weights,
                             layer_t* curr_layer,
                             float* results,
                             device_t* device) {
    auto session = get_session(device);
    auto op = std::make_unique<DepthwiseConvolution3dOp<dtype>>(
            curr_layer, NUM_TEST_CASES, session->cpu());
    if (session->empty()) {
        op->init(inputs, weights, results);
    } else {
        op->init(*session->last_op(), weights, results);
    }
    session->push_back(std::move(op));
}

void pointwise_convolution3d(float* inputs,
                             float* weights,
                             layer_t* curr_layer,
                             float* results,
                             device_t* device) {
    auto session = get_session(device);
    auto op = std::make_unique<PointwiseConvolution3dOp<dtype>>(
            curr_layer, NUM_TEST_CASES, session->cpu());
    if (session->empty()) {
        op->init(inputs, weights, results);
    } else {
        op->init(*session->last_op(), weights, results);
    }
    session->push_back(std::move(op));
}

}  // namespace nnet_mkl
