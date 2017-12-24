#ifndef _MKL_BATCH_NORM_H_
#define _MKL_BATCH_NORM_H_

#include "mkldnn.hpp"

#include "arch/nnet_mkl.h"
#include "core/nnet_fwd_defs.h"

namespace nnet_mkl {

template <typename DType>
class BatchNormOp : public BaseMklOp<DType> {
   public:
    // Weights are organized in blocks in this order.
    enum WeightsIndex {
        MeanIndex,
        VarianceIndex,
        ScaleshiftIndex,  // Gamma, then beta.
        NumWeightTypes
    };
    static constexpr DType kEpsilon = mkl_traits<DType>::to_type(1e-5);

    BatchNormOp(DType* input_buffer,
                DType* weights_buffer,
                DType* output_buffer,
                layer_t* _layer,
                int _batch_size,
                const mkldnn::engine& engine)
            : BaseMklOp<DType>(_layer, _batch_size, engine) {
        auto input_mem = create_input_memory(input_buffer);
        auto mean_mem = create_mean_memory(weights_buffer);
        auto variance_mem = create_variance_memory(weights_buffer);
        auto scaleshift_mem = create_scaleshift_memory(weights_buffer);
        auto output_mem = create_output_memory(output_buffer);

        INFO_MSG("BN, no input chaining\n");
        create_primitive(
                input_mem, mean_mem, variance_mem, scaleshift_mem, output_mem);
    }

   protected:
    // Returns true if the input to BN is the output of an FC layer.
    bool is_fc_output() {
        return this->layer->inputs.height == 1;
    }

    // Return a mem_dims object for the input, assuming nchw format.
    mem_dims get_input_dims() {
        int input_size =
                this->layer->inputs.rows * this->layer->inputs.height *
                (this->layer->inputs.cols + this->layer->inputs.align_pad);
        // The size of the batch normalization is indicated by the channel
        // dimension. Therefore, the output of a FC layer must have all its
        // dimensionality (except for batch size) compressed into the channel
        // dimension, with rows = cols = 1. Output of CONV layers can remain
        // the same.
        return is_fc_output() ? mem_dims({ this->batch_size, input_size, 1, 1 })
                              : mem_dims({ this->batch_size,
                                           this->layer->inputs.height,
                                           this->layer->inputs.rows,
                                           this->layer->inputs.cols });
    }

    // Return a mem_dims object for the output, assuming nchw format.
    mem_dims get_output_dims() {
        return get_input_dims();
    }

    // Return the dimensions of the means.
    mem_dims get_mean_dims() {
        int num_sets_weights = get_input_dims()[1];
        return { num_sets_weights };
    }

    // Return the dimensions of the variances.
    mem_dims get_variance_dims() {
        int num_sets_weights = get_input_dims()[1];
        return { num_sets_weights };
    }

    // Return the dimensions of the scaleshift weights.
    mem_dims get_scaleshift_dims() {
        int num_sets_weights = get_input_dims()[1];
        return{ 2, num_sets_weights };
    }

    // Create an input memory primitive from a raw pointer.
    mem_ref_t create_input_memory(DType* buffer) {
        return this->create_memory(buffer, get_input_dims(), mem_fmt::nchw);
    }

    // Create an output memory primitive from a raw pointer.
    mem_ref_t create_output_memory(DType* buffer) {
        return this->create_memory(
                buffer, get_output_dims(), mem_fmt::nchw, true);
    }

    // Create a memory primitive for the means.
    mem_ref_t create_mean_memory(DType* buffer) {
        auto mean_dims = get_mean_dims();
        return this->create_memory(
                buffer + MeanIndex * mean_dims[0], mean_dims, mem_fmt::x);
    }

    // Create a memory primitive for the variances.
    mem_ref_t create_variance_memory(DType* buffer) {
        auto variance_dims = get_variance_dims();
        return this->create_memory(buffer + VarianceIndex * variance_dims[0],
                                   variance_dims, mem_fmt::x);
    }

    // Create a memory primitive for the variances.
    mem_ref_t create_scaleshift_memory(DType* buffer) {
        auto scaleshift_dims = get_scaleshift_dims();
        return this->create_memory(
                buffer + ScaleshiftIndex * scaleshift_dims[1], scaleshift_dims,
                mem_fmt::nc);
    }

    // Create a batch norm primitive.
    mkldnn::primitive& create_primitive(mem_ref_t input,
                                        mem_ref_t mean,
                                        mem_ref_t variance,
                                        mem_ref_t scaleshift,
                                        mem_ref_t output) {
        const auto& input_md = input.get_primitive_desc().desc();
        auto bn_desc = mkldnn::batch_normalization_forward::desc(
                mkldnn::prop_kind::forward_inference,
                input_md,
                kEpsilon,
                mkldnn::use_global_stats | mkldnn::use_scale_shift);
        auto bn_pd = mkldnn::batch_normalization_forward::primitive_desc(
                bn_desc, this->engine);

        mem_ref_t bn_mean = this->reorder_input_if_needed(
                mean, bn_pd.mean_primitive_desc());
        mem_ref_t bn_variance = this->reorder_input_if_needed(
                variance, bn_pd.variance_primitive_desc());
        mem_ref_t bn_scaleshift = this->reorder_input_if_needed(
                scaleshift, bn_pd.weights_primitive_desc());

        // Must explicitly cast to primitive::at for batch_norm, or constructor
        // overload resolution actually picks the wrong constructor.
        typedef mkldnn::primitive::at at;
        this->template create_primitive_with_output_reorder<
                mkldnn::batch_normalization_forward>(
                bn_pd, output, at(input), at(bn_mean), at(bn_variance),
                at(bn_scaleshift));

        return this->worklist.back();
   }
};

void batch_norm(float* inputs,
                float* weights,
                layer_t* curr_layer,
                int batch_size,
                float* results,
                device_t* device);
}

#endif
