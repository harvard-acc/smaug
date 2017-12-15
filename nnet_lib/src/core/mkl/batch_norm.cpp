#include <iostream>

#include "core/mkl/batch_norm.h"
#include "utility/utility.h"

namespace nnet_mkl {

using namespace mkldnn;

// TODO: Move this into a common class shared by all the BN implementations.
enum {
    MEAN,
    VARIANCE,
    GAMMA,
    BETA
};

// Get the appropriate dimensions for BN if the output is from an FC layer.
memory::dims get_input_dims_for_fc_output(layer_t* curr_layer, int batch_size) {
    int input_size = get_input_activations_size(curr_layer) / NUM_TEST_CASES;
    return { batch_size, input_size, 1, 1 };
}

// Get the appropriate dimensions for BN if the output is from a CONV layer.
memory::dims get_input_dims_for_conv_output(layer_t* curr_layer,
                                            int batch_size) {
    return { batch_size, curr_layer->inputs.height, curr_layer->inputs.rows,
             curr_layer->inputs.cols };
}

void batch_norm(float* inputs,
                float* weights,
                layer_t* curr_layer,
                int batch_size,
                float* results,
                device_t* device) {
    std::vector<primitive> network;
    nnet_mkl::MklSession* session =
            reinterpret_cast<nnet_mkl::MklSession*>(device->session);

    bool is_fc_output = curr_layer->inputs.height == 1;
    memory::dims input_dims =
            is_fc_output
                    ? get_input_dims_for_fc_output(curr_layer, batch_size)
                    : get_input_dims_for_conv_output(curr_layer, batch_size);
    auto input_md =
            mem_d({ input_dims }, memory::data_type::f32, memory::format::nchw);
    auto input_mempd = mem_pd(input_md, session->cpu);
    auto input_memory = memory(input_mempd, inputs);

    // The channel dimension of the input indicates the number of parameter sets
    // (mean, variance, gamma, beta).
    int num_sets_weights =
            is_fc_output ? curr_layer->inputs.rows * curr_layer->inputs.cols
                         : curr_layer->inputs.height;
    ARRAY_2D(float, _weights, weights, num_sets_weights);

    memory::dims mean_var_dims = { num_sets_weights };
    auto mean_md = memory::desc(
            { mean_var_dims }, memory::data_type::f32, memory::format::x);
    auto mean_mem_pd = memory::primitive_desc(mean_md, session->cpu);
    auto mean_memory = memory(mean_mem_pd, &_weights[MEAN][0]);

    memory::dims var_dims = { num_sets_weights };
    auto var_md = memory::desc(
            { var_dims }, memory::data_type::f32, memory::format::x);
    auto var_mem_pd = memory::primitive_desc(var_md, session->cpu);
    auto var_memory = memory(var_mem_pd, &_weights[VARIANCE][0]);

    // First dimension is gamma, second is beta (the same way that we store them).
    memory::dims scaleshift_dims = { 2, num_sets_weights };
    auto scaleshift_md = memory::desc(
            { scaleshift_dims }, memory::data_type::f32, memory::format::nc);
    auto scaleshift_mem_pd = memory::primitive_desc(scaleshift_md, session->cpu);
    auto scaleshift_memory = memory(scaleshift_mem_pd, &_weights[GAMMA][0]);

    auto bn_desc = batch_normalization_forward::desc(
            prop_kind::forward_inference,
            input_md,
            1e-5,
            use_global_stats | use_scale_shift);
    auto bn_pd =
            batch_normalization_forward::primitive_desc(bn_desc, session->cpu);

    memory::dims output_dims = input_dims;
    auto output_md = mem_d(
            { output_dims }, memory::data_type::f32, memory::format::nchw);
    auto output_mem_pd = mem_pd(output_md, session->cpu);
    auto output_memory = memory(output_mem_pd, results);

    // Don't bother checking if transforms are needed, since the data is
    // guaranteed to arrive in a suitable format, and batch norm forwards is an
    // elementwise operation.

    auto bn_mean_memory = mean_memory;
    if (mem_pd(bn_pd.mean_primitive_desc()) != mean_mem_pd) {
        bn_mean_memory = memory(bn_pd.mean_primitive_desc());
        network.emplace_back(reorder(mean_memory, bn_mean_memory));
    }
    auto bn_var_memory = var_memory;
    if (mem_pd(bn_pd.variance_primitive_desc()) != var_mem_pd) {
        bn_var_memory = memory(bn_pd.variance_primitive_desc());
        network.emplace_back(reorder(var_memory, bn_var_memory));
    }
    auto bn_dst_memory = output_memory;
    if (mem_pd(bn_pd.dst_primitive_desc()) != output_mem_pd) {
        bn_dst_memory = memory(bn_pd.dst_primitive_desc());
        network.emplace_back(reorder(output_memory, bn_dst_memory));
    }
    auto bn_weights_memory = scaleshift_memory;
    if (mem_pd(bn_pd.weights_primitive_desc()) != scaleshift_mem_pd) {
        bn_weights_memory = memory(bn_pd.weights_primitive_desc());
        network.emplace_back(reorder(scaleshift_memory, bn_weights_memory));
    }

    network.emplace_back(
            batch_normalization_forward(bn_pd,
                                        primitive::at(input_memory),
                                        primitive::at(bn_mean_memory),
                                        primitive::at(bn_var_memory),
                                        primitive::at(bn_weights_memory),
                                        output_memory));

    stream(stream::kind::eager).submit(network).wait();
}

}  // namespace nnet_mkl
