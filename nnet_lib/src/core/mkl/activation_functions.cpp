#include <memory>
#include <vector>

#include "activation_functions.h"

namespace nnet_mkl {

using namespace mkldnn;

void sigmoid(float* activations,
             int batch_size,
             layer_t* layer,
             MklSession* session,
             float* results) {
    auto op =
            SIGMOID_IMPL == ExpUnit
                    ? std::make_unique<
                              mkl_impl::SigmoidActivationFunctionOp<dtype>>(
                              layer, batch_size, session->cpu())
                    : std::make_unique<lut::SigmoidActivationFunctionOp<dtype>>(
                              layer, batch_size, session->cpu());
    if (session->empty()) {
        op->init(activations, results);
    } else {
        op->init(*session->last_op(), results);
    }
    session->push_back(std::move(op));
}

void relu(float* activations,
          int batch_size,
          layer_t* layer,
          MklSession* session,
          float* results,
          float negative_slope) {
    auto op = std::make_unique<mkl_impl::ReluActivationFunctionOp<dtype>>(
            layer, batch_size, session->cpu());
    if (session->empty()) {
        op->init(activations, results, negative_slope);
    } else {
        op->init(*session->last_op(), results, negative_slope);
    }
    session->push_back(std::move(op));
}

void elu(float* activations,
         int batch_size,
         layer_t* layer,
         MklSession* session,
         float* results) {
    static const float alpha = 0.1;
    auto op = SIGMOID_IMPL == ExpUnit
                      ? std::make_unique<
                                mkl_impl::EluActivationFunctionOp<dtype>>(
                                layer, batch_size, session->cpu())
                      : std::make_unique<lut::EluActivationFunctionOp<dtype>>(
                                layer, batch_size, session->cpu());
    if (session->empty()) {
        op->init(activations, results, alpha);
    } else {
        op->init(*session->last_op(), results, alpha);
    }
    session->push_back(std::move(op));
}

void selu(float* activations,
          int batch_size,
          layer_t* layer,
          MklSession* session,
          float* results) {
    auto op = SIGMOID_IMPL == ExpUnit
                      ? std::make_unique<
                                mkl_impl::SeluActivationFunctionOp<dtype>>(
                                layer, batch_size, session->cpu())
                      : std::make_unique<lut::SeluActivationFunctionOp<dtype>>(
                                layer, batch_size, session->cpu());
    if (session->empty()) {
        op->init(activations, results);
    } else {
        op->init(*session->last_op(), results);
    }
    session->push_back(std::move(op));
}

void tanh(float* activations,
          int batch_size,
          layer_t* layer,
          MklSession* session,
          float* results) {
    auto op = SIGMOID_IMPL == ExpUnit
                      ? std::make_unique<
                                mkl_impl::TanhActivationFunctionOp<dtype>>(
                                layer, batch_size, session->cpu())
                      : std::make_unique<lut::TanhActivationFunctionOp<dtype>>(
                                layer, batch_size, session->cpu());
    if (session->empty()) {
        op->init(activations, results);
    } else {
        op->init(*session->last_op(), results);
    }
    session->push_back(std::move(op));
}

void softmax(float* activations,
             int batch_size,
             layer_t* layer,
             MklSession* session,
             float* results) {
    auto op = std::make_unique<mkl_impl::SoftmaxActivationFunctionOp<dtype>>(
            layer, batch_size, session->cpu());
    if (session->empty()) {
        op->init(activations, results);
    } else {
        op->init(*session->last_op(), results);
    }
    session->push_back(std::move(op));
}

void activation_fun(float* activations,
                    int batch_size,
                    layer_t* curr_layer,
                    float* results,
                    device_t* device) {
    auto session = get_session(device);
    activation_type function = curr_layer->activation;
    if (function == RELU) {
        relu(activations, batch_size, curr_layer, session, results, 0);
    } else if (function == SIGMOID) {
        sigmoid(activations, batch_size, curr_layer, session, results);
    } else if (function == LRELU) {
        static const float alpha = 0.1;
        relu(activations, batch_size, curr_layer, session, results, alpha);
    } else if (function == ELU) {
        elu(activations, batch_size, curr_layer, session, results);
    } else if (function == SELU) {
        selu(activations, batch_size, curr_layer, session, results);
    } else if (function == TANH) {
        tanh(activations, batch_size, curr_layer, session, results);
    } else if (function == SOFTMAX) {
        softmax(activations, batch_size, curr_layer, session, results);
    } else {
        assert(false && "This activation function is currently unsupported!");
    }
}

}  // namespace nnet_mkl
