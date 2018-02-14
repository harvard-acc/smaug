#ifndef _MKL_MATRIX_MULTIPLY_H_
#define _MKL_MATRIX_MULTIPLY_H_

#include "arch/nnet_mkl.h"
#include "core/nnet_fwd_defs.h"

namespace nnet_mkl {

template <typename DType>
class InnerProductOp : public BaseMklOp<DType> {
   public:
    using BaseMklOp<DType>::BaseMklOp;

    virtual void init(DType* input_buffer,
                      DType* weights_buffer,
                      DType* output_buffer) {
        prev_layer = nullptr;
        auto input_mem = create_input_memory(input_buffer);
        auto weight_mem = create_weight_memory(weights_buffer);
        auto bias_mem = create_bias_memory(weights_buffer);

        INFO_MSG("Fully connected\n");
        create_primitive(input_mem, weight_mem, bias_mem, output_buffer);
    }


    virtual void init(const BaseMklOp<DType>& prev_op,
                      DType* weights_buffer,
                      DType* output_buffer) {
        prev_layer = prev_op.get_layer();
        auto last_mem = prev_op.get_output_mem();
        // Batch norm layers have a requirement on the input dims which makes it
        // hard to chain with an upcoming FC layer, so we'll just skip chaining
        // for those.
        auto input_mem = (prev_layer && prev_layer->type == BATCH_NORM)
                                 ? create_input_memory(
                                           (float*)last_mem.get_data_handle())
                                 : last_mem;
        auto weight_mem = create_weight_memory(weights_buffer);
        auto bias_mem = create_bias_memory(weights_buffer);

        if (input_mem == last_mem)
            INFO_MSG("Fully connected, chaining\n");
        else
            INFO_MSG("Fully connected after BN, so no chaining\n");
        create_primitive(input_mem, weight_mem, bias_mem, output_buffer);
    }

    virtual std::string name() const { return "Inner product"; }

   protected:
    // Return a mem_dims object for the input, assuming nc format.
    mem_dims get_input_dims() {
        if (prev_layer && (prev_layer->type == CONV_STANDARD ||
                           prev_layer->type == CONV_DEPTHWISE ||
                           prev_layer->type == CONV_POINTWISE ||
                           prev_layer->type == POOLING)) {
            return { this->batch_size, prev_layer->outputs.height,
                     prev_layer->outputs.rows,
                     prev_layer->outputs.cols + prev_layer->outputs.align_pad };
        } else {
            return { this->batch_size,
                     this->layer->inputs.cols + this->layer->inputs.align_pad };
        }
    }

    // Return a mem_dims object for the output, assuming nc format.
    mem_dims get_output_dims() {
        return { this->batch_size, this->layer->outputs.cols };
    }

    // Return a mem_dims object for the weight.
    //
    // If the previous layer was a convolution or pooling layer, then we have
    // to provide 4 dimensions for the weights (aka unflattened weights);
    // otherwise, we provide 2.
    mem_dims get_weight_dims() {
        if (prev_layer && (prev_layer->type == CONV_STANDARD ||
                           prev_layer->type == CONV_DEPTHWISE ||
                           prev_layer->type == CONV_POINTWISE ||
                           prev_layer->type == POOLING)) {
            return { this->layer->weights.cols, prev_layer->outputs.height,
                     prev_layer->outputs.rows, prev_layer->outputs.cols };
        } else {
            return { this->layer->weights.cols, this->layer->weights.rows };
        }
    }

    // Return a mem_dims object for the bias, assuming x format.
    mem_dims get_bias_dims() {
        return { this->layer->biases.cols };
    }

    // Create an input memory primitive from a raw pointer.
    //
    // Returns the index to this primitive.
    mem_ref_t create_input_memory(DType* buffer) {
        mem_dims dims = get_input_dims();
        if (dims.size() == 4)
            return this->create_memory(buffer, dims, mem_fmt::nchw);
        else
            return this->create_memory(buffer, dims, mem_fmt::nc);
    }

    // Create an output memory primitive from a raw pointer.
    mem_ref_t create_output_memory(DType* buffer) {
        return this->create_memory(
                buffer, get_output_dims(), mem_fmt::nc, true);
    }

    // Create a weight memory primitive from a raw pointer.
    mem_ref_t create_weight_memory(DType* buffer) {
        mem_dims dims = get_input_dims();
        if (dims.size() == 4) {
            return this->create_memory(
                    buffer, get_weight_dims(), mem_fmt::oihw);
        } else {
            return this->create_memory(buffer, get_weight_dims(), mem_fmt::oi);
        }
    }

    // Create a bias memory primitive from a pointer, assuming nchw format.
    //
    // The pointer is assumed to point at the beginning of the weights, with
    // biases stored at the end.
    mem_ref_t create_bias_memory(DType* weights_buffer) {
        // Biases are assumed to not be transposed.
        float* biases = weights_buffer +
                        (this->layer->weights.rows * this->layer->weights.cols);
        return this->create_memory(biases, get_bias_dims(), mem_fmt::x);
    }

    // Create an inner product primitive.
    //
    // Supply the memory indices for inputs, weights, bias, and outputs.
    // Optionally, set force_output_format = true to force the output into nchw
    // format; otherwise, it will simply use whatever format MKL-DNN decides.
    mkldnn::primitive& create_primitive(mem_ref_t input,
                                        mem_ref_t weights,
                                        mem_ref_t bias,
                                        DType* output_buffer) {
        mem_dtype dtype = mkl_traits<DType>::dtype;
        auto mm_input_md = mem_d({ get_input_dims() }, dtype, mem_fmt::any);
        auto mm_weight_md = mem_d({ get_weight_dims() }, dtype, mem_fmt::any);
        auto mm_bias_md = mem_d({ get_bias_dims() }, dtype, mem_fmt::x);
        auto mm_output_md = mem_d({ get_output_dims() }, dtype, mem_fmt::any);

        auto mm_desc = mkldnn::inner_product_forward::desc(
                mkldnn::prop_kind::forward_inference, mm_input_md, mm_weight_md,
                mm_bias_md, mm_output_md);
        auto mm_pd = mkldnn::inner_product_forward::primitive_desc(
                mm_desc, this->engine);

        INFO_MSG("  FC input activations...\n");
        auto mm_inputs = this->reorder_input_if_needed(
                input, mm_pd.src_primitive_desc());
        INFO_MSG("  FC weights...\n");
        auto mm_weights = this->reorder_input_if_needed(
                weights, mm_pd.weights_primitive_desc());

        this->template create_primitive_no_output_reorder<
                mkldnn::inner_product_forward>(
                mm_pd, output_buffer, mm_inputs, mm_weights, bias);

        return this->worklist.back();
    }

    const layer_t* prev_layer;
};

void matrix_multiply_with_bias(float* inputs,
                               float* weights,
                               layer_t* curr_layer,
                               float* results,
                               device_t* device);

}  // namespace nnet_mkl

#endif

