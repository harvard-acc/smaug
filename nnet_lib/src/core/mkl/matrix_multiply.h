#ifndef _MKL_MATRIX_MULTIPLY_H_
#define _MKL_MATRIX_MULTIPLY_H_

#include "arch/nnet_mkl.h"

namespace nnet_mkl {

template <typename DType>
class InnerProductOp : public BaseMklOp<DType> {
   public:
    InnerProductOp(DType* input_buffer,
                   DType* weights_buffer,
                   DType* output_buffer,
                   layer_t* _layer,
                   int _batch_size,
                   mkldnn::engine& engine)
            : BaseMklOp<DType>(engine), layer(_layer), batch_size(_batch_size) {
        auto input_mem = create_input_memory(input_buffer);
        auto weight_mem = create_weight_memory(weights_buffer);
        auto bias_mem = create_bias_memory(weights_buffer);
        auto output_mem = create_output_memory(output_buffer);

        create_primitive(input_mem, weight_mem, bias_mem, output_mem);
    }

   protected:
    // Return a mem_dims object for the input, assuming nc format.
    mem_dims get_input_dims() {
        return { batch_size, layer->inputs.cols };
    }

    // Return a mem_dims object for the output, assuming nc format.
    mem_dims get_output_dims() {
        return { batch_size, layer->outputs.cols };
    }

    // Return a mem_dims object for the weight, assuming oi format.
    mem_dims get_weight_dims() {
        return { layer->weights.cols, layer->weights.rows - 1 };
    }

    // Return a mem_dims object for the bias, assuming x format.
    mem_dims get_bias_dims() { return { layer->weights.cols }; }

    // Create an input memory primitive from a raw pointer.
    //
    // Returns the index to this primitive.
    mem_ref_t create_input_memory(DType* buffer) {
        return this->create_memory(buffer, get_input_dims(), mem_fmt::nc);
    }

    // Create an output memory primitive from a raw pointer.
    mem_ref_t create_output_memory(DType* buffer) {
        return this->create_memory(
                buffer, get_output_dims(), mem_fmt::nc, true);
    }

    // Create a weight memory primitive from a raw pointer.
    mem_ref_t create_weight_memory(DType* buffer) {
        return this->create_memory(buffer, get_weight_dims(), mem_fmt::oi);
    }

    // Create a bias memory primitive from a pointer, assuming nchw format.
    //
    // The pointer is assumed to point at the beginning of the weights, with
    // biases stored at the end.
    mem_ref_t create_bias_memory(DType* weights_buffer) {
        // Biases are assumed to not be transposed.
        float* biases = weights_buffer + ((layer->weights.rows - 1) *
                                          layer->weights.cols);
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
                                        mem_ref_t output) {
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

        auto mm_inputs = this->reorder_input_if_needed(
                input, mm_pd.src_primitive_desc());
        auto mm_weights = this->reorder_input_if_needed(
                weights, mm_pd.weights_primitive_desc());

        this->template create_primitive_with_output_reorder<
                mkldnn::inner_product_forward>(
                mm_pd, output, mm_inputs, mm_weights, bias);

        return this->worklist.back();
    }

    // The inner product layer configuration.
    const layer_t* layer;
    const int batch_size;
};

void matrix_multiply_with_bias(float* inputs,
                               float* weights,
                               layer_t* curr_layer,
                               float* results,
                               device_t* device);

}  // namespace nnet_mkl

#endif

