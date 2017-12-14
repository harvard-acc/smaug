#include "activation_functions.h"

namespace nnet_mkl {

using namespace mkldnn;

void sigmoid(float* activations, int size, engine& cpu, float* results) {
    std::vector<primitive> network;

    memory::dims dims = { size };
    auto input_md = mem_d(
            { dims }, memory::data_type::f32, memory::format::x);
    auto input_mempd = mem_pd(input_md, cpu);
    auto input_memory = memory(input_mempd, activations);
    auto sigmoid_desc = eltwise_forward::desc(
            prop_kind::forward, algorithm::eltwise_logistic, input_md, 0, 0);
    auto sigmoid_pd = eltwise_forward::primitive_desc(sigmoid_desc, cpu);
    auto sigmoid_dst_memory = memory(sigmoid_pd.dst_primitive_desc(), results);
    network.emplace_back(
            eltwise_forward(sigmoid_pd, input_memory, sigmoid_dst_memory));
    stream(stream::kind::eager).submit(network).wait();
}

void relu(float* activations, int size, engine& cpu, float* results) {
    static const double negative_slope = 0;
    std::vector<primitive> network;

    memory::dims dims = { size };
    auto input_md = mem_d(
            { dims }, memory::data_type::f32, memory::format::x);
    auto input_mempd  = mem_pd(input_md, cpu);
    auto input_memory = memory(input_mempd, activations);
    auto output_md = mem_d(
            { dims }, memory::data_type::f32, memory::format::x);
    auto output_mempd  = mem_pd(output_md, cpu);
    auto output_memory = memory(output_mempd, results);
    auto relu_desc = eltwise_forward::desc(prop_kind::forward,
                                           algorithm::eltwise_relu,
                                           input_md,
                                           negative_slope);
    auto relu_pd = eltwise_forward::primitive_desc(relu_desc, cpu);
    network.emplace_back(
            eltwise_forward(relu_pd, input_memory, output_memory));
    stream(stream::kind::eager).submit(network).wait();
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
