#include <memory>
#include <vector>

#include "activation_functions.h"

namespace nnet_mkl {

using namespace mkldnn;

void sigmoid(float* activations, int size, engine& cpu, float* results) {
    SigmoidActivationFunctionOp sigmoid_op(activations, results, size, cpu);
    sigmoid_op.run();
}

void relu(float* activations,
          int size,
          engine& cpu,
          float* results,
          float negative_slope = 0) {
    ReluActivationFunctionOp relu_op(
            activations, results, size, cpu, negative_slope);
    relu_op.run();
}

void elu(float* activations, int size, engine& cpu, float* results) {
    static const float alpha = 0.1;
    EluActivationFunctionOp elu_op(activations, results, size, cpu, alpha);
    elu_op.run();
}

void selu(float* activations, int size, engine& cpu, float* results) {
    SeluActivationFunctionOp selu_op(activations, results, size, cpu);
    selu_op.run();
}

void tanh(float* activations, int size, engine& cpu, float* results) {
    TanhActivationFunctionOp tanh_op(activations, results, size, cpu);
    tanh_op.run();
}

void activation_fun(float* activations,
                    int size,
                    activation_type function,
                    float* results,
                    device_t* device) {
    nnet_mkl::MklSession* session =
            reinterpret_cast<nnet_mkl::MklSession*>(device->session);
    if (function == RELU) {
        relu(activations, size, session->cpu, results, 0);
    } else if (function == SIGMOID) {
        sigmoid(activations, size, session->cpu, results);
    } else if (function == LRELU) {
        static const float alpha = 0.1;
        relu(activations, size, session->cpu, results, alpha);
    } else if (function == ELU) {
        elu(activations, size, session->cpu, results);
    } else if (function == SELU) {
        selu(activations, size, session->cpu, results);
    } else if (function == TANH) {
        tanh(activations, size, session->cpu, results);
    } else {
        assert(false && "This activation function is currently unsupported!");
    }
}


}  // namespace nnet_mkl
