#include "mkldnn.hpp"

#include "pooling.h"

namespace nnet_mkl {

using namespace mkldnn;

void max_pooling_3d(float* inputs,
                    layer_t* curr_layer,
                    float* results,
                    device_t* device) {
    std::vector<primitive> network;
    nnet_mkl::MklSession* session =
            reinterpret_cast<nnet_mkl::MklSession*>(device->session);

    // Convert to MKL-DNN dimension descriptors.
    memory::dims pool_input_dims = { NUM_TEST_CASES, curr_layer->inputs.height,
                                     curr_layer->inputs.rows,
                                     curr_layer->inputs.cols };
    memory::dims pool_kernel = { curr_layer->weights.cols,
                                 curr_layer->weights.cols };
    memory::dims pool_strides = { curr_layer->field_stride,
                                  curr_layer->field_stride };
    memory::dims pool_padding = { 0, 0 };
    memory::dims pool_output_dims = { NUM_TEST_CASES,
                                      curr_layer->outputs.height,
                                      curr_layer->outputs.rows,
                                      curr_layer->outputs.cols };

    // Create memory descriptors.
    // Create primitive memory descriptors for user data.
    // Create memory primitives for user input data.
    auto user_input_md = mem_d(
            { pool_input_dims }, memory::data_type::f32, memory::format::nchw);
    auto user_input_memory_descriptor = mem_pd(user_input_md, session->cpu);
    auto user_input_memory = memory(user_input_memory_descriptor, inputs);

    // Create output memory descriptors.
    auto user_output_md = mem_d(
            { pool_output_dims }, memory::data_type::f32, memory::format::nchw);
    auto user_output_memory_descriptor = mem_pd(user_output_md, session->cpu);
    auto user_output_memory = memory(user_output_memory_descriptor, results);

    // Create memory descriptors for the pooling primitive.
    auto pool_output_md = mem_d(
            { pool_output_dims }, memory::data_type::f32, memory::format::any);

    // Create the pooling primitive descriptor.
    auto pool_desc = pooling_forward::desc(
            prop_kind::forward, algorithm::pooling_max, user_input_md,
            pool_output_md, pool_strides, pool_kernel, pool_padding,
            pool_padding, padding_kind::zero);
    auto pool_pd = pooling_forward::primitive_desc(pool_desc, session->cpu);

    // Pooling doesn't need input transformations...?

    // Create memory primitives for the output.
    auto pool_indices_memory = memory(pool_pd.workspace_primitive_desc());

    // Make a separate output memory for pooling if required.
    auto pool_output_memory = user_output_memory;
    if (mem_pd(pool_pd.dst_primitive_desc()) != user_output_memory_descriptor) {
        pool_output_memory = memory(pool_pd.dst_primitive_desc());
    }

    // Create the pooling primitive.
    network.emplace_back(pooling_forward(pool_pd, user_input_memory,
                                         pool_output_memory,
                                         pool_indices_memory));

    // Transform output if required.
    if (pool_output_memory != user_output_memory) {
        network.emplace_back(reorder(user_output_memory, pool_output_memory));
    }

    stream(stream::kind::eager).submit(network).wait();
}

} // namespace nnet_mkl
