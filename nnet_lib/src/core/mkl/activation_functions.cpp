#include <memory>
#include <vector>

#include "activation_functions.h"

namespace nnet_mkl {

using namespace mkldnn;

void sigmoid(float* activations, int size, engine& cpu, float* results) {
    SigmoidActivationFunctionOp sigmoid_op(activations, results, size, cpu);
    sigmoid_op.run();
}

void relu(float* activations, int size, engine& cpu, float* results) {
    ReluActivationFunctionOp relu_op(activations, results, size, cpu);
    relu_op.run();
}

void activation_fun(float* activations,
                    int size,
                    activation_type function,
                    float* results,
                    device_t* device) {
    nnet_mkl::MklSession* session =
            reinterpret_cast<nnet_mkl::MklSession*>(device->session);
    if (function == RELU) {
        relu(activations, size, session->cpu, results);
    } else if (function == SIGMOID) {
        sigmoid(activations, size, session->cpu, results);
    } else {
        assert(false && "This activation function is currently unsupported!");
    }
}


}  // namespace nnet_mkl
