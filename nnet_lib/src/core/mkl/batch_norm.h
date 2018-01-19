#ifndef _MKL_BATCH_NORM_H_
#define _MKL_BATCH_NORM_H_

#include "mkldnn.hpp"

#include "arch/nnet_mkl.h"
#include "core/nnet_fwd_defs.h"
#include "core/ref/batch_norm.h"

namespace nnet_mkl {

template <typename DType>
class BatchNormOp : public BaseMklOp<DType> {
   public:
    static constexpr DType kEpsilon = mkl_traits<DType>::to_type(1e-5);

    using BaseMklOp<DType>::BaseMklOp;

    virtual void init(DType* input_buffer,
                      DType* weights_buffer,
                      DType* output_buffer) {
        auto input_mem = create_input_memory(input_buffer);
        auto mean_mem = create_mean_memory(weights_buffer);
        auto variance_mem = create_variance_memory(weights_buffer);
        auto scaleshift_mem = create_scaleshift_memory(weights_buffer);
        auto output_mem = create_output_memory(output_buffer);

        INFO_MSG("BN, no input chaining\n");
        create_primitive(
                input_mem, mean_mem, variance_mem, scaleshift_mem, output_mem);
    }

    virtual void init(const BaseMklOp<DType>& prev_op,
                      DType* weights_buffer,
                      DType* output_buffer) {
        auto input_mem =
                is_fc_output()
                        ? create_input_memory((DType*)prev_op.get_output_mem()
                                                      .get_data_handle())
                        : prev_op.get_output_mem();
        auto mean_mem = create_mean_memory(weights_buffer);
        auto variance_mem = create_variance_memory(weights_buffer);
        auto scaleshift_mem = create_scaleshift_memory(weights_buffer);
        auto output_mem = create_output_memory(output_buffer);

        INFO_MSG("BN, chaining\n");
        create_primitive(
                input_mem, mean_mem, variance_mem, scaleshift_mem, output_mem);
    }

    virtual std::string name() const { return "Batch normalization"; }

   protected:
    // Returns true if the input to BN is the output of an FC layer.
    virtual bool is_fc_output() {
        return this->layer->inputs.height == 1;
    }

    // Return a mem_dims object for the input, assuming nchw format.
    virtual mem_dims get_input_dims() {
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
    virtual mem_dims get_output_dims() {
        return get_input_dims();
    }

    // Return the dimensions of the means.
    virtual mem_dims get_mean_dims() {
        int num_sets_weights = get_input_dims()[1];
        return { num_sets_weights };
    }

    // Return the dimensions of the variances.
    virtual mem_dims get_variance_dims() {
        int num_sets_weights = get_input_dims()[1];
        return { num_sets_weights };
    }

    // Return the dimensions of the scaleshift weights.
    virtual mem_dims get_scaleshift_dims() {
        int num_sets_weights = get_input_dims()[1];
        return{ 2, num_sets_weights };
    }

    // Create an input memory primitive from a raw pointer.
    virtual mem_ref_t create_input_memory(DType* buffer) {
        return this->create_memory(buffer, get_input_dims(), mem_fmt::nchw);
    }

    // Create an output memory primitive from a raw pointer.
    virtual mem_ref_t create_output_memory(DType* buffer) {
        return this->create_memory(
                buffer, get_output_dims(), mem_fmt::nchw, true);
    }

    // Create a memory primitive for the means.
    virtual mem_ref_t create_mean_memory(DType* buffer) {
        auto mean_dims = get_mean_dims();
        return this->create_memory(
                buffer + MeanIndex * mean_dims[0], mean_dims, mem_fmt::x);
    }

    // Create a memory primitive for the variances.
    virtual mem_ref_t create_variance_memory(DType* buffer) {
        auto variance_dims = get_variance_dims();
        return this->create_memory(buffer + VarianceIndex * variance_dims[0],
                                   variance_dims, mem_fmt::x);
    }

    // Create a memory primitive for the variances.
    virtual mem_ref_t create_scaleshift_memory(DType* buffer) {
        auto scaleshift_dims = get_scaleshift_dims();
        return this->create_memory(
                buffer + ScaleshiftIndex * scaleshift_dims[1], scaleshift_dims,
                mem_fmt::nc);
    }

    // Create a batch norm primitive.
    virtual mkldnn::primitive& create_primitive(mem_ref_t input,
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

// This batch norm operation assumes that we can precompute 1/sqrt(var + eps)
// during inference. This dramatically increases performance of batch norm.
//
// This operation cannot be implemented with the available MKL-DNN primitives
// (no element-wise operations), so we will call our own reference
// implementation.
template <typename DType>
class PrecomputedBatchNormOp : public BatchNormOp<DType> {
   protected:
    // Stores the raw pointers for the inputs, weights, and results that the
    // batch norm operation will need.
    struct BatchNormBuffers {
      DType* inputs;
      DType* weights;
      DType* results;
    };

   public:
    using BatchNormOp<DType>::BatchNormOp;

    virtual void init(DType* inputs_buffer,
                      DType* weights_buffer,
                      DType* outputs_buffer) {
        INFO_MSG("BN with precomputed weights.\n");
        auto input_mem = this->create_input_memory(inputs_buffer);
        auto output_mem = this->create_output_memory(outputs_buffer);
        create_primitive(input_mem, weights_buffer, output_mem, true);
    }

    virtual void init(const BaseMklOp<DType>& prev_op,
                      DType* weights_buffer,
                      DType* outputs_buffer) {
        INFO_MSG("BN with precomputed weights, chained.\n");
        auto input_mem = this->is_fc_output()
                                 ? this->create_input_memory(
                                           (DType*)prev_op.get_output_mem()
                                                   .get_data_handle())
                                 : prev_op.get_output_mem();
        auto output_mem = this->create_output_memory(outputs_buffer);
        create_primitive(input_mem, weights_buffer, output_mem, false);
    }

    virtual std::string name() const {
        return "Batch normalization (precomputed)";
    }

    virtual void run_work() {
        // First, run the input reorder if one is required.
        mkldnn::stream(mkldnn::stream::kind::eager).submit(this->worklist).wait();
        // Now run the batch norm.

        batch_norm_fxp(buffers.inputs, buffers.weights, this->layer,
                       this->batch_size, buffers.results);
    }

    // Since the computation is not done by an MKL-DNN, the back of the
    // worklist does not contain a valid result. Instead, directly return the
    // output memory.
    virtual const mkldnn::primitive& get_final_primitive() const {
        return this->get_output_mem();
    }

   protected:
    virtual void create_primitive(mem_ref_t input_mem,
                                  DType* weights,
                                  mem_ref_t output_mem,
                                  bool bypass_input_check) {
        if (!bypass_input_check) {
            auto expected_input_dims = this->get_input_dims();
            auto expected_input_md =
                    mem_d({ expected_input_dims }, mkl_traits<DType>::dtype,
                          mem_fmt::nchw);
            auto expected_input_pd = mem_pd(expected_input_md, this->engine);
            mem_ref_t bn_input =
                    this->reorder_input_if_needed(input_mem, expected_input_pd);
            buffers.inputs =
                    reinterpret_cast<DType*>(bn_input.get_data_handle());
        } else {
            buffers.inputs =
                    reinterpret_cast<DType*>(input_mem.get_data_handle());
        }

        buffers.weights = weights;
        buffers.results =
                reinterpret_cast<DType*>(output_mem.get_data_handle());
    }

    BatchNormBuffers buffers;
};

void batch_norm(float* inputs,
                float* weights,
                layer_t* curr_layer,
                int batch_size,
                float* results,
                device_t* device);
}

#endif
