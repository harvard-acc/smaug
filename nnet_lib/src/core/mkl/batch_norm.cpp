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
    auto op = std::make_unique<BatchNormOp<dtype>>(
            curr_layer, NUM_TEST_CASES, session->cpu());
    if (session->empty()) {
        op->init(inputs, weights, results);
    } else {
        op->init(*session->last_op(), weights, results);
    }
    session->push_back(std::move(op));
}

}  // namespace nnet_mkl
