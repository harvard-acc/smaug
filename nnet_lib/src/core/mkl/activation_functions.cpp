#include <memory>
#include <vector>

#include "activation_functions.h"

namespace nnet_mkl {

using namespace mkldnn;

BaseMklOpPtr sigmoid(float* activations,
                     int batch_size,
                     layer_t* layer,
                     MklSession* session,
                     float* results) {
    if (session->empty()) {
        return BaseMklOpPtr(new SigmoidActivationFunctionOp(
                activations, results, batch_size, layer, session->cpu()));
    } else {
        return BaseMklOpPtr(new SigmoidActivationFunctionOp(
                session->last_op(), results, batch_size, layer,
                session->cpu()));
    }
}

BaseMklOpPtr relu(float* activations,
                  int batch_size,
                  layer_t* layer,
                  MklSession* session,
                  float* results,
                  float negative_slope = 0) {
    if (session->empty()) {
        return BaseMklOpPtr(new ReluActivationFunctionOp(
                activations, results, batch_size, layer, session->cpu(),
                negative_slope));
    } else {
        return BaseMklOpPtr(new ReluActivationFunctionOp(
                session->last_op(), results, batch_size, layer, session->cpu(),
                negative_slope));
    }
}

BaseMklOpPtr elu(float* activations,
                 int batch_size,
                 layer_t* layer,
                 MklSession* session,
                 float* results) {
    static const float alpha = 0.1;
    if (session->empty()) {
        return BaseMklOpPtr(new EluActivationFunctionOp(activations, results,
                                                        batch_size, layer,
                                                        session->cpu(), alpha));
    } else {
        return BaseMklOpPtr(new EluActivationFunctionOp(
                session->last_op(), results, batch_size, layer, session->cpu(),
                alpha));
    }
}

BaseMklOpPtr selu(float* activations,
                  int batch_size,
                  layer_t* layer,
                  MklSession* session,
                  float* results) {
    if (session->empty()) {
        return BaseMklOpPtr(new SeluActivationFunctionOp(
                activations, results, batch_size, layer, session->cpu()));
    } else {
        return BaseMklOpPtr(new SeluActivationFunctionOp(
                session->last_op(), results, batch_size, layer,
                session->cpu()));
    }
}

BaseMklOpPtr tanh(float* activations,
                  int batch_size,
                  layer_t* layer,
                  MklSession* session,
                  float* results) {
    if (session->empty()) {
        return BaseMklOpPtr(new TanhActivationFunctionOp(
                activations, results, batch_size, layer, session->cpu()));
    } else {
        return BaseMklOpPtr(new TanhActivationFunctionOp(
                session->last_op(), results, batch_size, layer,
                session->cpu()));
    }
}

BaseMklOpPtr softmax(float* activations,
                     int batch_size,
                     layer_t* layer,
                     MklSession* session,
                     float* results) {
    if (session->empty()) {
        return BaseMklOpPtr(new SoftmaxActivationFunctionOp(
                activations, results, batch_size, layer, session->cpu()));
    } else {
        return BaseMklOpPtr(new SoftmaxActivationFunctionOp(session->last_op(),
                                                            results,
                                                            batch_size,
                                                            layer,
                                                            session->cpu()));
    }
}

void activation_fun(float* activations,
                    int batch_size,
                    layer_t* curr_layer,
                    float* results,
                    device_t* device) {
    auto session = get_session(device);
    activation_type function = curr_layer->activation;
    if (function == RELU) {
        session->add_op(
                relu(activations, batch_size, curr_layer, session, results, 0));
    } else if (function == SIGMOID) {
        session->add_op(
                sigmoid(activations, batch_size, curr_layer, session, results));
    } else if (function == LRELU) {
        static const float alpha = 0.1;
        session->add_op(relu(
                activations, batch_size, curr_layer, session, results, alpha));
    } else if (function == ELU) {
        session->add_op(
                elu(activations, batch_size, curr_layer, session, results));
    } else if (function == SELU) {
        session->add_op(
                selu(activations, batch_size, curr_layer, session, results));
    } else if (function == TANH) {
        session->add_op(
                tanh(activations, batch_size, curr_layer, session, results));
    } else if (function == SOFTMAX) {
        session->add_op(
                softmax(activations, batch_size, curr_layer, session, results));
    } else {
        assert(false && "This activation function is currently unsupported!");
    }
}

}  // namespace nnet_mkl
