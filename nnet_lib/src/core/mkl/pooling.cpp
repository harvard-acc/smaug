#include "mkldnn.hpp"

#include "core/mkl/pooling.h"

namespace nnet_mkl {

using namespace mkldnn;

void max_pooling_3d(float* inputs,
                    layer_t* curr_layer,
                    float* results,
                    device_t* device) {
    auto session = get_session(device);
    auto op = std::make_unique<MaxPoolingOp<dtype>>(
            curr_layer, NUM_TEST_CASES, session->cpu());
    if (session->empty()) {
        op->init(inputs, results);
    } else {
        op->init(*session->last_op(), results);
    }
    session->push_back(std::move(op));
}

void avg_pooling_3d(float* inputs,
                    layer_t* curr_layer,
                    float* results,
                    device_t* device) {
    auto session = get_session(device);
    auto op = std::make_unique<AvgPoolingOp<dtype>>(
            curr_layer, NUM_TEST_CASES, session->cpu());
    if (session->empty()) {
        op->init(inputs, results);
    } else {
        op->init(*session->last_op(), results);
    }
    session->push_back(std::move(op));
}

} // namespace nnet_mkl
