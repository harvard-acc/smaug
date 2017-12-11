#include <memory>

#include "mkldnn.hpp"

#include "core/nnet_fwd_defs.h"
#include "arch/nnet_mkl.h"

namespace nnet_mkl {

using namespace mkldnn;

void convolution3d(float* inputs,
                   float* weights,
                   layer_t* curr_layer,
                   float* results,
                   device_t* device) {
    std::vector<primitive> network;
    nnet_mkl::MklSession* session =
            reinterpret_cast<nnet_mkl::MklSession*>(device->session);

    // TODO: We don't actually have any biases yet!
    float* biases = new float[curr_layer->outputs.height];
    for (int i = 0; i < curr_layer->outputs.height; i++)
        biases[i] = 0;

    int c_padding = curr_layer->c_padding;
    // Convert to MKL-DNN dimension descriptors.
    memory::dims conv_input_dims = { NUM_TEST_CASES, curr_layer->inputs.height,
                                     curr_layer->inputs.rows - 2 * c_padding,
                                     curr_layer->inputs.cols - 2 * c_padding };

    memory::dims conv_weight_dims = { curr_layer->outputs.height,
                                      curr_layer->weights.height,
                                      curr_layer->weights.rows,
                                      curr_layer->weights.cols };

    memory::dims conv_bias_dims = { curr_layer->outputs.height };

    memory::dims conv_output_dims = { NUM_TEST_CASES,
                                      curr_layer->outputs.height,
                                      curr_layer->outputs.rows,
                                      curr_layer->outputs.cols };

    memory::dims conv_stride = { curr_layer->field_stride,
                                 curr_layer->field_stride };

    // We pass this twice...?
    auto conv_padding = { curr_layer->c_padding, curr_layer->c_padding };

    // Create memory descriptors for the user data (using the actual data layout).
    auto user_input_md = mem_d(
            { conv_input_dims }, memory::data_type::f32, memory::format::nchw);
    auto user_weight_md = mem_d(
            { conv_weight_dims }, memory::data_type::f32, memory::format::oihw);
    auto user_bias_md = mem_d(
            { conv_bias_dims }, memory::data_type::f32, memory::format::x);
    auto user_output_md = mem_d(
            { conv_output_dims }, memory::data_type::f32, memory::format::nchw);

    // Create primitive memory descriptors for user data.
    auto user_input_memory_descriptor = mem_pd(user_input_md, session->cpu);
    auto user_weight_memory_descriptor = mem_pd(user_weight_md, session->cpu);
    auto user_bias_memory_descriptor = mem_pd(user_bias_md, session->cpu);
    auto user_output_memory_descriptor = mem_pd(user_output_md, session->cpu);

    // Create memory primitives for user input data (input, weights, biases).
    auto user_input_memory = memory(user_input_memory_descriptor, inputs);
    auto user_weight_memory = memory(user_weight_memory_descriptor, weights);
    auto user_bias_memory = memory(user_bias_memory_descriptor, biases);

    // Create memory descriptors for the convolution primitive.
    auto conv_input_md = mem_d(
            { conv_input_dims }, memory::data_type::f32, memory::format::any);
    auto conv_weight_md = mem_d(
            { conv_weight_dims }, memory::data_type::f32, memory::format::any);
    auto conv_bias_md = mem_d(
            { conv_bias_dims }, memory::data_type::f32, memory::format::any);
    auto conv_output_md = mem_d(
            { conv_output_dims }, memory::data_type::f32, memory::format::any);

    // Create the convolution primitive descriptor.
    auto conv_desc = convolution_forward::desc(
            prop_kind::forward, algorithm::convolution_direct, conv_input_md,
            conv_weight_md, conv_bias_md, conv_output_md, conv_stride,
            conv_padding, conv_padding, mkldnn::padding_kind::zero);
    auto conv_pd = convolution_forward::primitive_desc(conv_desc, session->cpu);

    // Check if a data layout transform is required.
    auto conv_input_memory = user_input_memory;
    if (mem_pd(conv_pd.src_primitive_desc()) !=
        user_input_memory_descriptor) {
        conv_input_memory = memory(conv_pd.src_primitive_desc());
        network.emplace_back(reorder(user_input_memory, conv_input_memory));
    }

    auto conv_weight_memory = user_weight_memory;
    if (mem_pd(conv_pd.weights_primitive_desc()) !=
        user_weight_memory_descriptor) {
        conv_weight_memory = memory(conv_pd.weights_primitive_desc());
        network.emplace_back(reorder(user_weight_memory, conv_weight_memory));
    }

    // Create memory primitives for the output.
    bool output_needs_reorder = mem_pd(conv_pd.dst_primitive_desc()) !=
                                user_output_memory_descriptor;
    auto conv_output_memory =
            output_needs_reorder
                    ? memory(conv_pd.dst_primitive_desc())
                    : memory(user_output_memory_descriptor, results);

    // Finally, create the convolution primitive.
    network.emplace_back(
            convolution_forward(conv_pd, conv_input_memory, conv_weight_memory,
                                user_bias_memory, conv_output_memory));

    // TODO: If the output format is not nchw, we need to convert it.
    // Eventually we should just stick with one data format as much as we can.
    auto user_output_memory = conv_output_memory;
    if (output_needs_reorder) {
        user_output_memory = memory(user_output_memory_descriptor, results);
        network.emplace_back(reorder(conv_output_memory, user_output_memory));
    }

    stream(stream::kind::eager).submit(network).wait();

    // TODO: If the output format is not nchw, we need to convert it.
    // Eventually we'll just stick with one data format as much as we can.

    delete[] biases;
}

}  // namespace nnet_mkl
