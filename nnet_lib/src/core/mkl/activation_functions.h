#ifndef _MKL_ACTIVATION_FUNCTIONS_H_
#define _MKL_ACTIVATION_FUNCTIONS_H_

#include "mkldnn.hpp"

#include "arch/nnet_mkl.h"
#include "utility/utility.h"

namespace nnet_mkl {

template <typename DType>
class ActivationFunctionOp : public BaseMklOp<DType> {
   public:
    ActivationFunctionOp(layer_t* layer,
                         int batch_size,
                         const mkldnn::engine& eng)
            : BaseMklOp<DType>(layer, batch_size, eng),
              input_size(get_dims_size(&layer->outputs)) {}
    virtual ~ActivationFunctionOp() {}

   protected:
    mem_ref_t create_memory(DType* buffer, int size, bool is_output = false) {
        return BaseMklOp<DType>::create_memory(
                buffer, mem_dims({ size }), mem_fmt::x, is_output);
    }

    // Create a primitive descriptor for this eltwise primitive.
    mkldnn::eltwise_forward::primitive_desc create_primitive_desc(
            mkldnn::algorithm alg,
            mem_d input_md,
            DType alpha = 0,
            DType beta = 0) {
        auto desc =
                mkldnn::eltwise_forward::desc(mkldnn::prop_kind::forward,
                                              alg,
                                              input_md,
                                              alpha,
                                              beta);
        return mkldnn::eltwise_forward::primitive_desc(desc, this->engine);
    }

    // Return a reference to the eltwise primitive.
    mkldnn::eltwise_forward& create_primitive(mkldnn::algorithm alg,
                                              mem_ref_t input,
                                              mem_ref_t output,
                                              DType alpha = 0,
                                              DType beta = 0) {
        auto pd = create_primitive_desc(
                alg, input.get_primitive_desc().desc(), alpha, beta);
        this->worklist.emplace_back(mkldnn::eltwise_forward(pd, input, output));
        return static_cast<mkldnn::eltwise_forward&>(this->worklist.back());
    }

    mkldnn::eltwise_forward& create_primitive(mkldnn::algorithm alg,
                                              const mkldnn::primitive& input_prim,
                                              mem_d input_md,
                                              DType* output_buffer,
                                              DType alpha = 0,
                                              DType beta = 0) {
        auto pd = create_primitive_desc(alg, input_md, alpha, beta);
        mem_ref_t output = BaseMklOp<dtype>::create_memory(
                pd.dst_primitive_desc(), output_buffer, true);
        this->worklist.emplace_back(mkldnn::eltwise_forward(pd, input_prim, output));
        return static_cast<mkldnn::eltwise_forward&>(this->worklist.back());
    }

    int input_size;
};

class ReluActivationFunctionOp : public ActivationFunctionOp<dtype> {
   public:
    ReluActivationFunctionOp(dtype* input_buffer,
                             dtype* output_buffer,
                             int batch_size,
                             layer_t* layer,
                             const mkldnn::engine& engine,
                             dtype _negative_slope = 0)
            : ActivationFunctionOp(layer, batch_size, engine),
              negative_slope(_negative_slope) {
        INFO_MSG("RELU\n");
        auto input_mem = create_memory(input_buffer, this->input_size);
        auto output_mem = create_memory(output_buffer, this->input_size, true);
        create_primitive(mkldnn::algorithm::eltwise_relu,
                         input_mem,
                         output_mem,
                         negative_slope);
    }

    ReluActivationFunctionOp(const BaseMklOpPtr& prev_op,
                             dtype* output_buffer,
                             int batch_size,
                             layer_t* layer,
                             const mkldnn::engine& engine,
                             dtype _negative_slope = 0)
            : ActivationFunctionOp(layer, batch_size, engine),
              negative_slope(_negative_slope) {
        INFO_MSG("RELU, chaining\n");
        create_primitive(mkldnn::algorithm::eltwise_relu,
                         prev_op->get_final_primitive(),
                         prev_op->get_output_mem_desc(),
                         output_buffer,
                         negative_slope);
    }

    virtual ~ReluActivationFunctionOp() {}
    virtual std::string name() const {
        return negative_slope == 0 ? "RELU" : "LRELU";
    }

   protected:
    dtype negative_slope;
};

class SigmoidActivationFunctionOp : public ActivationFunctionOp<dtype> {
   public:
    SigmoidActivationFunctionOp(dtype* input_buffer,
                                dtype* output_buffer,
                                int batch_size,
                                layer_t* layer,
                                const mkldnn::engine& engine)
            : ActivationFunctionOp(layer, batch_size, engine) {
        auto input_mem = create_memory(input_buffer, this->input_size);
        auto output_mem = create_memory(output_buffer, this->input_size, true);
        INFO_MSG("Sigmoid\n");
        create_primitive(
                mkldnn::algorithm::eltwise_logistic, input_mem, output_mem);
    }

    SigmoidActivationFunctionOp(const BaseMklOpPtr& prev_op,
                                dtype* output_buffer,
                                int batch_size,
                                layer_t* layer,
                                const mkldnn::engine& engine)
            : ActivationFunctionOp(layer, batch_size, engine) {
        INFO_MSG("Sigmoid, chaining\n");
        create_primitive(mkldnn::algorithm::eltwise_logistic,
                         prev_op->get_final_primitive(),
                         prev_op->get_output_mem_desc(),
                         output_buffer);
    }
    virtual ~SigmoidActivationFunctionOp() {}
    virtual std::string name() const { return "Sigmoid"; }
};

class EluActivationFunctionOp : public ActivationFunctionOp<dtype> {
   public:
    EluActivationFunctionOp(
            dtype* input_buffer,
            dtype* output_buffer,
            int batch_size,
            layer_t* layer,
            const mkldnn::engine& engine,
            dtype negative_slope = mkl_traits<dtype>::to_type(0.1))
            : ActivationFunctionOp(layer, batch_size, engine) {
        auto input_mem = create_memory(input_buffer, this->input_size);
        auto output_mem = create_memory(output_buffer, this->input_size, true);
        INFO_MSG("ELU\n");
        create_primitive(mkldnn::algorithm::eltwise_elu,
                         input_mem,
                         output_mem,
                         negative_slope);
    }
    EluActivationFunctionOp(
            const BaseMklOpPtr& prev_op,
            dtype* output_buffer,
            int batch_size,
            layer_t* layer,
            const mkldnn::engine& engine,
            dtype negative_slope = mkl_traits<dtype>::to_type(0.1))
            : ActivationFunctionOp(layer, batch_size, engine) {
        INFO_MSG("ELU, chaining\n");
        create_primitive(mkldnn::algorithm::eltwise_elu,
                         prev_op->get_final_primitive(),
                         prev_op->get_output_mem_desc(),
                         output_buffer,
                         negative_slope);
    }
    virtual ~EluActivationFunctionOp() {}
    virtual std::string name() const { return "ELU"; }
};

class TanhActivationFunctionOp : public ActivationFunctionOp<dtype> {
   public:
    TanhActivationFunctionOp(dtype* input_buffer,
                             dtype* output_buffer,
                             int batch_size,
                             layer_t* layer,
                             const mkldnn::engine& engine)
            : ActivationFunctionOp(layer, batch_size, engine) {
        auto input_mem = create_memory(input_buffer, this->input_size);
        auto output_mem = create_memory(output_buffer, this->input_size, true);
        INFO_MSG("Tanh\n");
        create_primitive(
                mkldnn::algorithm::eltwise_tanh, input_mem, output_mem);
    }

    TanhActivationFunctionOp(const BaseMklOpPtr& prev_op,
                             dtype* output_buffer,
                             int batch_size,
                             layer_t* layer,
                             const mkldnn::engine& engine)
            : ActivationFunctionOp(layer, batch_size, engine) {
        INFO_MSG("Tanh, chaining\n");
        create_primitive(mkldnn::algorithm::eltwise_tanh,
                         prev_op->get_final_primitive(),
                         prev_op->get_output_mem_desc(),
                         output_buffer);
    }
    virtual ~TanhActivationFunctionOp() {}
    virtual std::string name() const { return "Tanh"; }
};

class SeluActivationFunctionOp : public ActivationFunctionOp<dtype> {
   public:
    static constexpr dtype alpha = mkl_traits<dtype>::to_type(1.6733);
    static constexpr dtype lambda = mkl_traits<dtype>::to_type(1.0507);

    SeluActivationFunctionOp(dtype* input_buffer,
                             dtype* output_buffer,
                             int batch_size,
                             layer_t* layer,
                             const mkldnn::engine& engine)
            : ActivationFunctionOp(layer, batch_size, engine) {
        INFO_MSG("SELU\n");
        auto input_mem = create_memory(input_buffer, this->input_size);
        auto intermediate_mem = create_memory(nullptr, this->input_size);
        auto output_mem = create_memory(output_buffer, this->input_size);
        // SELU can be implemented using an ELU, followed by a scaling.
        create_primitive(mkldnn::algorithm::eltwise_elu,
                         input_mem,
                         intermediate_mem,
                         alpha);
        // The linear algorithm performs y = Ax + B.
        create_primitive(mkldnn::algorithm::eltwise_linear,
                         intermediate_mem,
                         output_mem,
                         lambda);
    }

    SeluActivationFunctionOp(const BaseMklOpPtr& prev_op,
                             dtype* output_buffer,
                             int batch_size,
                             layer_t* layer,
                             const mkldnn::engine& engine)
            : ActivationFunctionOp(layer, batch_size, engine) {
        INFO_MSG("SELU, chaining\n");
        auto elu_pd = this->create_primitive_desc(
                mkldnn::algorithm::eltwise_elu, prev_op->get_output_mem_desc(),
                alpha);
        auto intermediate_mem =
                BaseMklOp<dtype>::create_memory(elu_pd.dst_primitive_desc());

        auto linear_pd = this->create_primitive_desc(
                mkldnn::algorithm::eltwise_linear,
                elu_pd.dst_primitive_desc().desc(), lambda);
        auto output_mem = BaseMklOp<dtype>::create_memory(
                linear_pd.dst_primitive_desc(), output_buffer, true);

        this->worklist.emplace_back(mkldnn::eltwise_forward(
                elu_pd, prev_op->get_final_primitive(), intermediate_mem));
        this->worklist.emplace_back(mkldnn::eltwise_forward(
                linear_pd, this->worklist.back(), output_mem));
    }

    virtual ~SeluActivationFunctionOp() {}
    virtual std::string name() const { return "SELU"; }
};

class SoftmaxActivationFunctionOp : public ActivationFunctionOp<dtype> {
   public:
    SoftmaxActivationFunctionOp(dtype* input_buffer,
                                dtype* output_buffer,
                                int batch_size,
                                layer_t* layer,
                                const mkldnn::engine& engine)
            : ActivationFunctionOp(layer, batch_size, engine) {
        INFO_MSG("Softmax\n");
        auto input_mem =
                create_memory(input_buffer, batch_size, this->input_size);
        auto output_mem = create_memory(
                output_buffer, this->input_size, batch_size, true);
        create_primitive(input_mem, output_mem);
    }

    SoftmaxActivationFunctionOp(const BaseMklOpPtr& prev_op,
                                dtype* output_buffer,
                                int batch_size,
                                layer_t* layer,
                                const mkldnn::engine& engine)
            : ActivationFunctionOp(layer, batch_size, engine) {
        INFO_MSG("Softmax, chaining\n");
        auto output_mem = create_memory(
                output_buffer, batch_size, this->input_size, true);
        create_primitive(prev_op->get_final_primitive(),
                         prev_op->get_output_mem_desc(), output_mem);
    }

    mem_ref_t create_memory(dtype* buffer,
                            int batch_size,
                            int softmax_size,
                            bool is_output = false) {
        return BaseMklOp<dtype>::create_memory(
                buffer,
                mem_dims({ batch_size, softmax_size }),
                mem_fmt::nc,
                is_output);
    }

    mkldnn::softmax_forward& create_primitive(mem_ref_t input,
                                              mem_ref_t output) {
        auto desc = mkldnn::softmax_forward::desc(
                mkldnn::prop_kind::forward_inference,
                input.get_primitive_desc().desc(),
                1);
        auto pd = mkldnn::softmax_forward::primitive_desc(desc, engine);
        worklist.emplace_back(
                mkldnn::softmax_forward(pd, input, output));
        return static_cast<mkldnn::softmax_forward&>(worklist.back());
    }

    mkldnn::softmax_forward& create_primitive(const mkldnn::primitive& input,
                                              mem_d input_md,
                                              mem_ref_t output) {
        auto desc = mkldnn::softmax_forward::desc(
                mkldnn::prop_kind::forward_inference, input_md, 1);
        auto pd = mkldnn::softmax_forward::primitive_desc(desc, engine);
        worklist.emplace_back(mkldnn::softmax_forward(pd, input, output));
        return static_cast<mkldnn::softmax_forward&>(worklist.back());
    }

    virtual ~SoftmaxActivationFunctionOp() {}
    virtual std::string name() const { return "Softmax"; }
};

BaseMklOpPtr sigmoid(float* activations,
                     int batch_size,
                     layer_t* layer,
                     MklSession* session,
                     float* results);
BaseMklOpPtr relu(float* activations,
                  int batch_size,
                  layer_t* layer,
                  MklSession* session,
                  float* results);
BaseMklOpPtr elu(float* activations,
                 int batch_size,
                 layer_t* layer,
                 MklSession* session,
                 float* results);
BaseMklOpPtr selu(float* activations,
                  int batch_size,
                  layer_t* layer,
                  MklSession* session,
                  float* results);
BaseMklOpPtr tanh(float* activations,
                  int batch_size,
                  layer_t* layer,
                  MklSession* session,
                  float* results);
BaseMklOpPtr softmax(float* a,
                     int batch_size,
                     layer_t* layer,
                     MklSession* session,
                     float* results);

void activation_fun(float* activations,
                    int batch_size,
                    layer_t* layer,
                    float* results,
                    device_t* device);

}  // namespace nnet_mkl

#endif
