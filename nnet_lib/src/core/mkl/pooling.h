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
                 mkldnn::engine& engine)
            : BaseMklOp<DType>(engine), layer(_layer), batch_size(_batch_size) {
        auto input_mem = create_input_memory(input_buffer);
        auto output_mem = create_output_memory(output_buffer);

        create_primitive(input_mem, output_mem);
    }

   protected:
    // Return a mem_dims object for the input, assuming nchw format.
    mem_dims get_input_dims() {
        return { batch_size, layer->inputs.height, layer->inputs.rows,
                 layer->inputs.cols };
    }

    // Return a mem_dims object for the output, assuming nchw format.
    mem_dims get_output_dims() {
        return { batch_size, layer->outputs.height, layer->outputs.rows,
                 layer->outputs.cols };
    }

    // Return a mem_dims object for the pooling dims.
    mem_dims get_pool_dims() {
        return { layer->weights.cols, layer->weights.cols };
    }

    // Return a mem_dims object for the pooling strides.
    mem_dims get_pool_strides() {
        return { layer->field_stride, layer->field_stride };
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

    // Create an inner product primitive.
    //
    // Supply the memory indices for inputs, weights, bias, and outputs.
    // Optionally, set force_output_format = true to force the output into nchw
    // format; otherwise, it will simply use whatever format MKL-DNN decides.
    mkldnn::primitive& create_primitive(mem_ref_t input, mem_ref_t output) {
        auto pool_output_md = mem_d(
                { get_output_dims() }, mkl_traits<DType>::dtype, mem_fmt::any);

        // Create the pooling primitive descriptor.
        auto pool_desc = mkldnn::pooling_forward::desc(
                mkldnn::prop_kind::forward, mkldnn::algorithm::pooling_max,
                input.get_primitive_desc().desc(), pool_output_md,
                get_pool_strides(), get_pool_dims(), get_pool_padding(),
                get_pool_padding(), mkldnn::padding_kind::zero);
        auto pool_pd = mkldnn::pooling_forward::primitive_desc(
                pool_desc, this->engine);

        // Pooling requires a separate workspace memory.
        // There seems to be a bug in MKL-DNN where we can't not use the
        // workspace (or an exception gets thrown).
        mem_ref_t workspace =
                this->create_memory(pool_pd.workspace_primitive_desc());

        // The output reorder needs to be delayed until after the operation
        // primitive.
        mkldnn::memory temp_output = output;
        bool reordered = false;
        if (this->needs_reorder(output, pool_pd.dst_primitive_desc())) {
            temp_output = this->create_memory(pool_pd.dst_primitive_desc());
            reordered = true;
        }

        this->worklist.emplace_back(mkldnn::pooling_forward(
                pool_pd, input, temp_output, workspace));

        if (reordered) {
            this->worklist.emplace_back(mkldnn::reorder(temp_output, output));
        }

        return this->worklist.back();
    }

   // The pooling layer configuration.
   const layer_t* layer;
   const int batch_size;
};

void max_pooling_3d(float* inputs,
                    layer_t* curr_layer,
                    float* result,
                    device_t* device);

}  // namespace nnet_mkl

#endif
