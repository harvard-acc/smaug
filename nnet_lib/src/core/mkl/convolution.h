#ifndef _MKL_CONVOLUTION_H_
#define _MKL_CONVOLUTION_H_

#include "arch/nnet_mkl.h"
#include "core/nnet_fwd_defs.h"

namespace nnet_mkl {

template <typename DType>
class Convolution3dOp : public BaseMklOp<DType> {
  public:
   Convolution3dOp(DType* input_buffer,
                   DType* weights_buffer,
                   DType* output_buffer,
                   layer_t* _layer,
                   int _batch_size,
                   const mkldnn::engine& engine)
           : BaseMklOp<DType>(_layer, _batch_size, engine) {
       auto input_mem = create_input_memory(input_buffer);
       auto weight_mem = create_weight_memory(weights_buffer);
       auto bias_mem = create_bias_memory();

       INFO_MSG("Convolution\n");
       create_primitive(input_mem, weight_mem, bias_mem, output_buffer);
    }

    Convolution3dOp(const BaseMklOpPtr& prev_op,
                    DType* weights_buffer,
                    DType* output_buffer,
                    layer_t* _layer,
                    int _batch_size,
                    const mkldnn::engine& engine)
            : BaseMklOp<DType>(_layer, _batch_size, engine) {
        auto input_mem = prev_op->get_output_mem();
        auto weight_mem = create_weight_memory(weights_buffer);
        auto bias_mem = create_bias_memory();

        INFO_MSG("Convolution, chaining\n");
        create_primitive(input_mem, weight_mem, bias_mem, output_buffer);
    }

    virtual std::string name() const { return "Convolution"; }

   protected:
    // Return a mem_dims object for the input, assuming nchw format.
    mem_dims get_input_dims() {
        int c_padding = this->layer->c_padding;
        return { this->batch_size, this->layer->inputs.height,
                 this->layer->inputs.rows - 2 * c_padding,
                 this->layer->inputs.cols - 2 * c_padding };
    }

    // Return a mem_dims object for the output, assuming nchw format.
    mem_dims get_output_dims() {
        return { this->batch_size, this->layer->outputs.height,
                 this->layer->outputs.rows, this->layer->outputs.cols };
    }

    // Return a mem_dims object for the weight, assuming oihw format.
    mem_dims get_weight_dims() {
        return { this->layer->outputs.height, this->layer->weights.height,
                 this->layer->weights.rows, this->layer->weights.cols };
    }

    // Return a mem_dims object for the bias, assuming x format.
    mem_dims get_bias_dims() {
        return { this->layer->outputs.height };
    }

    // Create an input memory primitive from a pointer, assuming nchw format.
    //
    // Returns the index to this primitive.
    mem_ref_t create_input_memory(DType* buffer) {
        return this->create_memory(buffer, get_input_dims(), mem_fmt::nchw);
    }

    // Create an output memory primitive from a pointer, assuming nchw format.
    mem_ref_t create_output_memory(DType* buffer) {
        return this->create_memory(
                buffer, get_output_dims(), mem_fmt::nchw, true);
    }

    // Create a weight memory primitive from a pointer, assuming oihw format.
    mem_ref_t create_weight_memory(DType* buffer) {
        return this->create_memory(buffer, get_weight_dims(), mem_fmt::oihw);
    }

    // Create a bias memory primitive from a pointer, assuming nchw format.
    mem_ref_t create_bias_memory() {
        biases = std::unique_ptr<DType[]>(
                new DType[this->layer->outputs.height]);
        for (int i = 0; i < this->layer->outputs.height; i++)
            biases[i] = 0;
        return this->create_memory(biases.get(), get_bias_dims(), mem_fmt::x);
    }

    // Create a convolution primitive.
    //
    // Supply the memory indices for inputs, weights, bias, and outputs.
    // If the output format is not mem_fmt::any, then the output will be
    // reordered if necessary into that specified format.
    mkldnn::primitive& create_primitive(mem_ref_t inputs,
                                        mem_ref_t weights,
                                        mem_ref_t bias,
                                        DType* output_buffer) {
        mem_dtype dtype = mkl_traits<DType>::dtype;
        auto conv_input_md = mem_d({ get_input_dims() }, dtype, mem_fmt::any);
        auto conv_weight_md = mem_d({ get_weight_dims() }, dtype, mem_fmt::any);
        auto conv_bias_md = mem_d({ get_bias_dims() }, dtype, mem_fmt::any);
        auto conv_output_md = mem_d({ get_output_dims() }, dtype, mem_fmt::any);

        mem_dims conv_stride = { this->layer->field_stride,
                                 this->layer->field_stride };
        // We pass this twice...?
        mem_dims conv_padding = { this->layer->c_padding,
                                  this->layer->c_padding };

        auto conv_desc = mkldnn::convolution_forward::desc(
                mkldnn::prop_kind::forward,
                mkldnn::algorithm::convolution_direct, conv_input_md,
                conv_weight_md, conv_bias_md, conv_output_md, conv_stride,
                conv_padding, conv_padding, mkldnn::padding_kind::zero);
        auto conv_pd = mkldnn::convolution_forward::primitive_desc(
                conv_desc, this->engine);

        // Inputs can be eagerly reordered if required.
        INFO_MSG("  Conv input activations...\n");
        mem_ref_t conv_input = this->reorder_input_if_needed(
                inputs, conv_pd.src_primitive_desc());
        INFO_MSG("  Conv weights...\n");
        mem_ref_t conv_weights = this->reorder_input_if_needed(
                weights, conv_pd.weights_primitive_desc());

        this->template create_primitive_no_output_reorder<
                mkldnn::convolution_forward>(
                conv_pd, output_buffer, conv_input, conv_weights, bias);

        return this->worklist.back();
    }

    // TODO: We don't actually have biases in the weights yet!
    std::unique_ptr<DType[]> biases;
};

// TODO: Share code between this and the standard convolution. This will
// require some significant refactoring to avoid calling virtual functions from
// within the constructor.
template <typename DType>
class DepthwiseConvolution3dOp : public BaseMklOp<DType> {
   public:
    DepthwiseConvolution3dOp(DType* input_buffer,
                             DType* weights_buffer,
                             DType* output_buffer,
                             layer_t* _layer,
                             int _batch_size,
                             const mkldnn::engine& engine)
            : BaseMklOp<DType>(_layer, _batch_size, engine) {
        auto input_mem = create_input_memory(input_buffer);
        auto weight_mem = create_weight_memory(weights_buffer);
        auto bias_mem = create_bias_memory();

        INFO_MSG("Depthwise convolution\n");
        create_primitive(input_mem, weight_mem, bias_mem, output_buffer);
    }

    DepthwiseConvolution3dOp(const BaseMklOpPtr& prev_op,
                             DType* weights_buffer,
                             DType* output_buffer,
                             layer_t* _layer,
                             int _batch_size,
                             const mkldnn::engine& engine)
            : BaseMklOp<DType>(_layer, _batch_size, engine) {
        auto input_mem = prev_op->get_output_mem();
        auto weight_mem = create_weight_memory(weights_buffer);
        auto bias_mem = create_bias_memory();

        INFO_MSG("Depthwise convolution, chaining\n");
        create_primitive(input_mem, weight_mem, bias_mem, output_buffer);
    }

    virtual std::string name() const { return "Depthwise convolution"; }

   protected:
    // Return a mem_dims object for the input, assuming nchw format.
    mem_dims get_input_dims() {
        int c_padding = this->layer->c_padding;
        return { this->batch_size, this->layer->inputs.height,
                 this->layer->inputs.rows - 2 * c_padding,
                 this->layer->inputs.cols - 2 * c_padding };
    }

    // Return a mem_dims object for the output, assuming nchw format.
    mem_dims get_output_dims() {
        return { this->batch_size, this->layer->outputs.height,
                 this->layer->outputs.rows, this->layer->outputs.cols };
    }

    // Return a mem_dims object for the weight, assuming goihw format.
    mem_dims get_weight_dims() {
        return { this->layer->outputs.height, 1, 1, this->layer->weights.rows,
                 this->layer->weights.cols };
    }

    // Return a mem_dims object for the bias, assuming x format.
    mem_dims get_bias_dims() {
        return { this->layer->outputs.height };
    }

    // Create an input memory primitive from a pointer, assuming nchw format.
    //
    // Returns the index to this primitive.
    mem_ref_t create_input_memory(DType* buffer) {
        return this->create_memory(buffer, get_input_dims(), mem_fmt::nchw);
    }

    // Create an output memory primitive from a pointer, assuming nchw format.
    mem_ref_t create_output_memory(DType* buffer) {
        return this->create_memory(
                buffer, get_output_dims(), mem_fmt::nchw, true);
    }

    // Create a weight memory primitive from a pointer, assuming goihw format.
    mem_ref_t create_weight_memory(DType* buffer) {
        return this->create_memory(buffer, get_weight_dims(), mem_fmt::goihw);
    }

    // Create a bias memory primitive from a pointer, assuming nchw format.
    mem_ref_t create_bias_memory() {
        biases = std::unique_ptr<DType[]>(
                new DType[this->layer->outputs.height]);
        for (int i = 0; i < this->layer->outputs.height; i++)
            biases[i] = 0;
        return this->create_memory(biases.get(), get_bias_dims(), mem_fmt::x);
    }

    // Create a convolution primitive.
    //
    // Supply the memory indices for inputs, weights, bias, and outputs.
    // If the output format is not mem_fmt::any, then the output will be
    // reordered if necessary into that specified format.
    mkldnn::primitive& create_primitive(mem_ref_t inputs,
                                        mem_ref_t weights,
                                        mem_ref_t bias,
                                        DType* output_buffer) {
        mem_dtype dtype = mkl_traits<DType>::dtype;
        auto conv_input_md = mem_d({ get_input_dims() }, dtype, mem_fmt::any);
        auto conv_weight_md = mem_d({ get_weight_dims() }, dtype, mem_fmt::any);
        auto conv_bias_md = mem_d({ get_bias_dims() }, dtype, mem_fmt::any);
        auto conv_output_md = mem_d({ get_output_dims() }, dtype, mem_fmt::any);

        mem_dims conv_stride = { this->layer->field_stride,
                                 this->layer->field_stride };
        // We pass this twice...?
        mem_dims conv_padding = { this->layer->c_padding,
                                  this->layer->c_padding };

        auto conv_desc = mkldnn::convolution_forward::desc(
                mkldnn::prop_kind::forward,
                mkldnn::algorithm::convolution_direct, conv_input_md,
                conv_weight_md, conv_bias_md, conv_output_md, conv_stride,
                conv_padding, conv_padding, mkldnn::padding_kind::zero);
        auto conv_pd = mkldnn::convolution_forward::primitive_desc(
                conv_desc, this->engine);

        // Inputs can be eagerly reordered if required.
        INFO_MSG("  Depthwise conv input activations...\n");
        mem_ref_t conv_input = this->reorder_input_if_needed(
                inputs, conv_pd.src_primitive_desc());
        INFO_MSG("  Depthwise conv weights...\n");
        mem_ref_t conv_weights = this->reorder_input_if_needed(
                weights, conv_pd.weights_primitive_desc());

        this->template create_primitive_no_output_reorder<
                mkldnn::convolution_forward>(
                conv_pd, output_buffer, conv_input, conv_weights, bias);

        return this->worklist.back();
    }

    // TODO: We don't actually have biases in the weights yet!
    std::unique_ptr<DType[]> biases;
};

void convolution3d(float* inputs,
                   float* weights,
                   layer_t* curr_layer,
                   float* results,
                   device_t* device);

void depthwise_convolution3d(float* inputs,
                             float* weights,
                             layer_t* curr_layer,
                             float* results,
                             device_t* device);

}  // namespace nnet_mkl

#endif
