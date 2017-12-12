#include <memory>

#include "mkldnn.hpp"

#include "core/nnet_fwd_defs.h"
#include "arch/nnet_mkl.h"

namespace nnet_mkl {

using namespace mkldnn;

void matrix_multiply_with_bias(float* inputs,
                               float* weights,
                               layer_t* curr_layer,
                               float* results,
                               device_t* device) {
    std::vector<primitive> network;
    nnet_mkl::MklSession* session =
            reinterpret_cast<nnet_mkl::MklSession*>(device->session);

    // Convert to MKL-DNN dimension descriptors.
    memory::dims mm_input_dims = { NUM_TEST_CASES,
                                   curr_layer->inputs.cols };
    // NOTE: The weights not only need to be transposed, but the actual
    // dimensions have to be specified as the transpose too.
    memory::dims mm_weight_dims = { curr_layer->weights.cols,
                                    curr_layer->weights.rows - 1 };
    memory::dims mm_bias_dims = { curr_layer->weights.cols };
    memory::dims mm_output_dims = { NUM_TEST_CASES,
                                    curr_layer->outputs.cols };

    // Weights are assumed to not be transposed.
    float* biases = weights +
             ((curr_layer->weights.rows - 1) * curr_layer->weights.cols);

    // Create memory descriptors.
    auto user_input_md = mem_d(
            { mm_input_dims }, memory::data_type::f32, memory::format::nc);
    auto user_weight_md = mem_d(
            { mm_weight_dims }, memory::data_type::f32, memory::format::oi);
    auto user_bias_md = mem_d(
            { mm_bias_dims }, memory::data_type::f32, memory::format::x);
    auto user_output_md = mem_d(
            { mm_output_dims }, memory::data_type::f32, memory::format::nc);

    // Create primitive memory descriptors for user data.
    auto user_input_memory_descriptor = mem_pd(user_input_md, session->cpu);
    auto user_weight_memory_descriptor = mem_pd(user_weight_md, session->cpu);
    auto user_bias_memory_descriptor = mem_pd(user_bias_md, session->cpu);
    auto user_output_memory_descriptor = mem_pd(user_output_md, session->cpu);

    // Create memory primitives for user input data (input, weights, biases).
    auto user_input_memory = memory(user_input_memory_descriptor, inputs);
    auto user_weight_memory = memory(user_weight_memory_descriptor, weights);
    auto user_bias_memory = memory(user_bias_memory_descriptor, biases);

    // Create memory descriptors for the inner product primitive.
    auto mm_input_md = mem_d(
            { mm_input_dims }, memory::data_type::f32, memory::format::any);
    auto mm_weight_md = mem_d(
            { mm_weight_dims }, memory::data_type::f32, memory::format::any);
    auto mm_bias_md = mem_d(
            { mm_bias_dims }, memory::data_type::f32, memory::format::any);
    auto mm_output_md = mem_d(
            { mm_output_dims }, memory::data_type::f32, memory::format::any);

    // Create the inner product primitive descriptor.
    auto mm_desc = inner_product_forward::desc(prop_kind::forward_inference,
                                               mm_input_md, mm_weight_md,
                                               user_bias_md, mm_output_md);
    auto mm_pd = inner_product_forward::primitive_desc(mm_desc, session->cpu);

    // Check if a data layout transform is required.
    auto mm_input_memory = user_input_memory;
    if (mem_pd(mm_pd.src_primitive_desc()) != user_input_memory_descriptor) {
        mm_input_memory = memory(mm_pd.src_primitive_desc());
        network.emplace_back(reorder(user_input_memory, mm_input_memory));
    }

    auto mm_weight_memory = user_weight_memory;
    if (mem_pd(mm_pd.weights_primitive_desc()) !=
        user_weight_memory_descriptor) {
        mm_weight_memory = memory(mm_pd.src_primitive_desc());
        network.emplace_back(reorder(user_weight_memory, mm_weight_memory));
    }

    // Create memory primitives for the output.
    bool output_fmt_matches = (mem_pd(mm_pd.dst_primitive_desc()) !=
                               user_output_memory_descriptor);
    auto mm_output_memory =
            output_fmt_matches ? memory(user_output_memory_descriptor, results)
                               : memory(mm_pd.dst_primitive_desc());

    // Finally, create the inner product primitive.
    network.emplace_back(
            inner_product_forward(mm_pd, mm_input_memory, mm_weight_memory,
                                  user_bias_memory, mm_output_memory));

    auto user_output_memory = mm_output_memory;
    if (!output_fmt_matches) {
        user_output_memory = memory(user_output_memory_descriptor, results);
        network.emplace_back(reorder(mm_output_memory, user_output_memory));
    }

    stream(stream::kind::eager).submit(network).wait();
}

}  // namespace nnet_mkl
