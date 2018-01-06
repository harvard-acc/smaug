#ifndef _MKL_POOLING_H_
#define _MKL_POOLING_H_

#include "core/nnet_fwd_defs.h"
#include "arch/nnet_mkl.h"
#include "utility/utility.h"

namespace nnet_mkl {

// PoolingType is of type enum pool_type.
template <typename DType, int PoolingType>
class PoolingOp : public BaseMklOp<DType> {
   public:
    PoolingOp(DType* input_buffer,
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

    PoolingOp(const BaseMklOpPtr& prev_op,
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

    virtual std::string name() const {
        return PoolingType == MAX ? "Max pooling" : "Average pooling";
    }

   protected:
    int compute_output_size(int size, int field_size, int stride) {
        return (size - field_size) / stride + 1;
    }

    // Return a mem_dims object for the input, assuming nchw format.
    mem_dims get_input_dims() {
        return { this->batch_size, this->layer->inputs.height,
                 this->layer->inputs.rows,
                 this->layer->inputs.cols + this->layer->inputs.align_pad };
    }

    // Return a mem_dims object for the output, assuming nchw format.
    //
    // If the input to the MKL pooling operation is padded to a data alignment,
    // then we have to make sure the output column size is with respect to this
    // PADDED amount, or the pooling primitive will raise an error.
    mem_dims get_output_dims() {
        int output_cols = compute_output_size(
                this->layer->inputs.cols + this->layer->inputs.align_pad,
                this->layer->weights.cols,
                this->layer->field_stride);
        return { this->batch_size, this->layer->outputs.height,
                 this->layer->outputs.rows, output_cols };
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
    // Use the right padding to accomplish output alignment padding.
    mem_dims get_pool_left_padding() {
        if (this->layer->c_padding != 0) {
            fprintf(stderr,
                    "Warning: The MKL pooling operation ignores padding!\n");
        }
        return { 0, 0 };
    }

    // Determine how much additional padding is required on the input columns
    // to meet data alignment requirements on the output.
    mem_dims get_pool_right_padding() {
        // If the output doesn't require any alignment, then we're done.
        if (this->layer->outputs.align_pad == 0)
            return { 0, 0 };

        // If the output width, after accounting for the input padding, still
        // doesn't need padding, then we're also done.
        int output_cols = get_output_dims()[3];
        int add_output_pad = calc_padding(output_cols, DATA_ALIGNMENT);
        if (add_output_pad == 0)
            return { 0, 0 };

        // Calculate how many more input pixels would be needed to produce
        // add_output_pad more output pixels. That is the additional input
        // padding.
        int add_input_pad = (add_output_pad - 1) * this->layer->field_stride +
                            this->layer->weights.cols;
        return {0, add_input_pad};
    }

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
        mkldnn::algorithm alg = PoolingType == MAX
                                        ? mkldnn::algorithm::pooling_max
                                        : mkldnn::algorithm::pooling_avg;
        return mkldnn::pooling_forward::desc(
                mkldnn::prop_kind::forward_inference, alg, input_md,
                pool_output_md, get_pool_strides(), get_pool_dims(),
                get_pool_left_padding(), get_pool_right_padding(),
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

template <typename DType>
using MaxPoolingOp = PoolingOp<DType, MAX>;

template <typename DType>
using AvgPoolingOp = PoolingOp<DType, AVG>;

void max_pooling_3d(float* inputs,
                    layer_t* curr_layer,
                    float* result,
                    device_t* device);

void avg_pooling_3d(float* inputs,
                    layer_t* curr_layer,
                    float* result,
                    device_t* device);

}  // namespace nnet_mkl

#endif
