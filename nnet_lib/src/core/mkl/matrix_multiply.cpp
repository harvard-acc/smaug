#include <memory>

#include "mkldnn.hpp"

#include "core/mkl/matrix_multiply.h"
#include "core/nnet_fwd_defs.h"
#include "arch/nnet_mkl.h"

namespace nnet_mkl {

using namespace mkldnn;

void matrix_multiply_with_bias(float* inputs,
                               float* weights,
                               layer_t* curr_layer,
                               float* results,
                               device_t* device) {
    auto session = get_session(device);
    session->oplist.emplace_back(new InnerProductOp<dtype>(inputs,
                                                           weights,
                                                           results,
                                                           curr_layer,
                                                           NUM_TEST_CASES,
                                                           session->cpu));
}

}  // namespace nnet_mkl
