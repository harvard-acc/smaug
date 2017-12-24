#ifndef _MKL_POOLING_H_
#define _MKL_POOLING_H_

#include "core/nnet_fwd_defs.h"
#include "arch/nnet_mkl.h"

namespace nnet_mkl {

template <typename DType>
class MaxPoolingOp : public BaseMklOp<DType> {
   public:
    MaxPoolingOp(DType* input_buffer,
                 DType* output_buffer,
                 layer_t* _layer,
                 int _batch_size,
                 const mkldnn::engine& engine)
            : BaseMklOp<DType>(_layer, _batch_size, engine) {
        auto input_mem = create_input_memory(input_buffer);
        auto output_mem = create_output_memory(output_buffer);

        INFO_MSG("Pooling\n");
        create_primitive(input_mem, output_mem);
    }

    MaxPoolingOp(const BaseMklOpPtr& prev_op,
                 DType* output_buffer,
                 layer_t* _layer,
                 int _batch_size,
                 const mkldnn::engine& engine)
            : BaseMklOp<DType>(_layer, _batch_size, engine) {
        INFO_MSG("Pooling, chaining\n");
        create_primitive(prev_op->get_final_primitive(),
                         prev_op->get_output_mem_desc(),
                         output_buffer);
    }

   protected:
    // Return a mem_dims object for the input, assuming nchw format.
    mem_dims get_input_dims() {
        return { this->batch_size, this->layer->inputs.height,
                 this->layer->inputs.rows, this->layer->inputs.cols };
    }

    // Return a mem_dims object for the output, assuming nchw format.
    mem_dims get_output_dims() {
        return { this->batch_size, this->layer->outputs.height,
                 this->layer->outputs.rows, this->layer->outputs.cols };
    }

    // Return a mem_dims object for the pooling dims.
    mem_dims get_pool_dims() {
        return { this->layer->weights.cols, this->layer->weights.cols };
    }

    // Return a mem_dims object for the pooling strides.
    mem_dims get_pool_strides() {
        return { this->layer->field_stride, this->layer->field_stride };
    }

    // Return a mem_dims object for the pooling padding.
    mem_dims get_pool_padding() { return { 0, 0 }; }

    // Create an input memory primitive from a raw pointer.
    //
    // Returns the index to this primitive.
    mem_ref_t create_input_memory(DType* buffer) {
        return this->create_memory(buffer, get_input_dims(), mem_fmt::nchw);
    }

    // Create an output memory primitive from a raw pointer.
    mem_ref_t create_output_memory(DType* buffer) {
        return this->create_memory(
                buffer, get_output_dims(), mem_fmt::nchw, true);
    }

    // Create a pooling descriptor.
    mkldnn::pooling_forward::desc create_pooling_desc(mem_d input_md) {
        auto pool_output_md = mem_d(
                { get_output_dims() }, mkl_traits<DType>::dtype, mem_fmt::any);
        return mkldnn::pooling_forward::desc(
                mkldnn::prop_kind::forward_inference, mkldnn::algorithm::pooling_max,
                input_md, pool_output_md, get_pool_strides(), get_pool_dims(),
                get_pool_padding(), get_pool_padding(),
                mkldnn::padding_kind::zero);
    }

    // Create a pooling primitive.
    //
    // This overload assumes that the input/output memory primitive have
    // already been created, so the pooling primitive will reorder the output
    // into that format.
    mkldnn::primitive& create_primitive(mem_ref_t input, mem_ref_t output) {
        auto pool_desc = create_pooling_desc(input.get_primitive_desc().desc());
        auto pool_pd = mkldnn::pooling_forward::primitive_desc(
                pool_desc, this->engine);

        this->template create_primitive_with_output_reorder<
                mkldnn::pooling_forward>(pool_pd, output, input);
        return this->worklist.back();
    }

    // Create a pooling primitive.
    //
    // This overload takes as input the output of another (e.g. previous)
    // primitive, and it stores the output in the format specified by the
    // pooling primitive.
    mkldnn::primitive& create_primitive(const mkldnn::primitive& input_prim,
                                        mem_d input_mem_d,
                                        DType* output_buffer) {
        auto pool_desc = create_pooling_desc(input_mem_d);
        auto pool_pd = mkldnn::pooling_forward::primitive_desc(
                pool_desc, this->engine);

        this->template create_primitive_no_output_reorder<
                mkldnn::pooling_forward>(pool_pd, output_buffer, input_prim);
        return this->worklist.back();
    }
};

void max_pooling_3d(float* inputs,
                    layer_t* curr_layer,
                    float* result,
                    device_t* device);

}  // namespace nnet_mkl

#endif
