#include <memory>
#include <vector>

#include "activation_functions.h"

namespace nnet_mkl {

using namespace mkldnn;

BaseMklOpPtr sigmoid(float* activations,
                     int size,
                     MklSession* session,
                     float* results) {
    if (session->empty()) {
        return BaseMklOpPtr(new SigmoidActivationFunctionOp(
                activations, results, size, session->cpu));
    } else {
        return BaseMklOpPtr(new SigmoidActivationFunctionOp(
                session->last_op(), results, size, session->cpu));
    }
}

BaseMklOpPtr relu(float* activations,
                  int size,
                  MklSession* session,
                  float* results,
                  float negative_slope = 0) {
    if (session->empty()) {
        return BaseMklOpPtr(new ReluActivationFunctionOp(
                activations, results, size, session->cpu, negative_slope));
    } else {
        return BaseMklOpPtr(new ReluActivationFunctionOp(session->last_op(),
                                                         results,
                                                         size,
                                                         session->cpu,
                                                         negative_slope));
    }
}

BaseMklOpPtr elu(float* activations,
                 int size,
                 MklSession* session,
                 float* results) {
    static const float alpha = 0.1;
    if (session->empty()) {
        return BaseMklOpPtr(new EluActivationFunctionOp(
                activations, results, size, session->cpu, alpha));
    } else {
        return BaseMklOpPtr(new EluActivationFunctionOp(
                session->last_op(), results, size, session->cpu, alpha));
    }
}

BaseMklOpPtr selu(float* activations,
                  int size,
                  MklSession* session,
                  float* results) {
    if (session->empty()) {
        return BaseMklOpPtr(new SeluActivationFunctionOp(
                activations, results, size, session->cpu));
    } else {
        return BaseMklOpPtr(new SeluActivationFunctionOp(
                session->last_op(), results, size, session->cpu));
    }
}

BaseMklOpPtr tanh(float* activations,
                  int size,
                  MklSession* session,
                  float* results) {
    if (session->empty()) {
        return BaseMklOpPtr(new TanhActivationFunctionOp(
                activations, results, size, session->cpu));
    } else {
        return BaseMklOpPtr(new TanhActivationFunctionOp(
                session->last_op(), results, size, session->cpu));
    }
}

BaseMklOpPtr softmax(float* activations,
                     int batch_size,
                     int softmax_size,
                     MklSession* session,
                     float* results) {
    if (session->empty()) {
        return BaseMklOpPtr(new SoftmaxActivationFunctionOp(
                activations, results, batch_size, softmax_size, session->cpu));
    } else {
        return BaseMklOpPtr(new SoftmaxActivationFunctionOp(session->last_op(),
                                                            results,
                                                            batch_size,
                                                            softmax_size,
                                                            session->cpu));
    }
}

void activation_fun(float* activations,
                    int batch_size,
                    int input_size,
                    activation_type function,
                    float* results,
                    device_t* device) {
    auto session = get_session(device);
    // Most of these functions are element-wise, so they don't need to know
    // about batches.
    int total_size = batch_size * input_size;
    if (function == RELU) {
        session->oplist.emplace_back(
                relu(activations, total_size, session, results, 0));
    } else if (function == SIGMOID) {
        session->oplist.emplace_back(
                sigmoid(activations, total_size, session, results));
    } else if (function == LRELU) {
        static const float alpha = 0.1;
        session->oplist.emplace_back(
                relu(activations, total_size, session, results, alpha));
    } else if (function == ELU) {
        session->oplist.emplace_back(
                elu(activations, total_size, session, results));
    } else if (function == SELU) {
        session->oplist.emplace_back(
                selu(activations, total_size, session, results));
    } else if (function == TANH) {
        session->oplist.emplace_back(
                tanh(activations, total_size, session, results));
    } else if (function == SOFTMAX) {
        session->oplist.emplace_back(softmax(
                activations, batch_size, input_size, session, results));
    } else {
        assert(false && "This activation function is currently unsupported!");
    }
}

}  // namespace nnet_mkl
