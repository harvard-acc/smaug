#include "mkldnn.hpp"

#include "core/mkl/pooling.h"

namespace nnet_mkl {

using namespace mkldnn;

void max_pooling_3d(float* inputs,
                    layer_t* curr_layer,
                    float* results,
                    device_t* device) {
    auto session = get_session(device);
    if (session->empty()) {
        session->add_op(new MaxPoolingOp<dtype>(
                inputs, results, curr_layer, NUM_TEST_CASES, session->cpu()));
    } else {
        session->add_op(new MaxPoolingOp<dtype>(session->last_op(),
                                                results,
                                                curr_layer,
                                                NUM_TEST_CASES,
                                                session->cpu()));
    }
}

void avg_pooling_3d(float* inputs,
                    layer_t* curr_layer,
                    float* results,
                    device_t* device) {
    auto session = get_session(device);
    if (session->empty()) {
        session->add_op(new AvgPoolingOp<dtype>(
                inputs, results, curr_layer, NUM_TEST_CASES, session->cpu()));
    } else {
        session->add_op(new AvgPoolingOp<dtype>(session->last_op(),
                                                results,
                                                curr_layer,
                                                NUM_TEST_CASES,
                                                session->cpu()));
    }
}

} // namespace nnet_mkl
