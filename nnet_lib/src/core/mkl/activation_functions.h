#ifndef _MKL_ACTIVATION_FUNCTIONS_H_
#define _MKL_ACTIVATION_FUNCTIONS_H_

#include "mkldnn.hpp"

#include "arch/nnet_mkl.h"

namespace nnet_mkl {

template <typename DType>
class ActivationFunctionOp : public BaseMklOp<DType> {
   public:
    ActivationFunctionOp(mkldnn::engine& eng) : BaseMklOp<DType>(eng) {}
    virtual ~ActivationFunctionOp() {}

   protected:
    mem_ref_t create_memory(DType* buffer, int size, bool is_output = false) {
        return BaseMklOp<DType>::create_memory(
                buffer, mem_dims({ size }), mem_fmt::x, is_output);
    }

    // Create a return a reference to the operation primitive.
    mkldnn::eltwise_forward& create_primitive(mkldnn::algorithm alg,
                                              mem_ref_t input,
                                              mem_ref_t output,
                                              DType alpha = 0,
                                              DType beta = 0) {
        auto desc =
                mkldnn::eltwise_forward::desc(mkldnn::prop_kind::forward,
                                              alg,
                                              input.get_primitive_desc().desc(),
                                              alpha,
                                              beta);
        auto pd = mkldnn::eltwise_forward::primitive_desc(desc, this->engine);
        this->worklist.emplace_back(mkldnn::eltwise_forward(pd, input, output));
        return static_cast<mkldnn::eltwise_forward&>(this->worklist.back());
    }
};

class ReluActivationFunctionOp : public ActivationFunctionOp<dtype> {
   public:
    ReluActivationFunctionOp(dtype* input_buffer,
                             dtype* output_buffer,
                             int size,
                             mkldnn::engine& engine,
                             dtype negative_slope = 0)
            : ActivationFunctionOp(engine) {
        auto input_mem_index = create_memory(input_buffer, size);
        auto output_mem_index = create_memory(output_buffer, size, true);
        create_primitive(mkldnn::algorithm::eltwise_relu,
                         input_mem_index,
                         output_mem_index,
                         negative_slope);
    }
    virtual ~ReluActivationFunctionOp() {}
};

class SigmoidActivationFunctionOp : public ActivationFunctionOp<dtype> {
   public:
    SigmoidActivationFunctionOp(dtype* input_buffer,
                                dtype* output_buffer,
                                int size,
                                mkldnn::engine& engine)
            : ActivationFunctionOp(engine) {
        auto input_mem_index = create_memory(input_buffer, size);
        auto output_mem_index = create_memory(output_buffer, size, true);
        create_primitive(mkldnn::algorithm::eltwise_logistic,
                         input_mem_index,
                         output_mem_index);
    }
    virtual ~SigmoidActivationFunctionOp() {}
};

class EluActivationFunctionOp : public ActivationFunctionOp<dtype> {
   public:
    EluActivationFunctionOp(
            dtype* input_buffer,
            dtype* output_buffer,
            int size,
            mkldnn::engine& engine,
            dtype negative_slope = mkl_traits<dtype>::to_type(0.1))
            : ActivationFunctionOp(engine) {
        auto input_mem_index = create_memory(input_buffer, size);
        auto output_mem_index = create_memory(output_buffer, size, true);
        create_primitive(mkldnn::algorithm::eltwise_elu,
                         input_mem_index,
                         output_mem_index,
                         negative_slope);
    }
    virtual ~EluActivationFunctionOp() {}
};

class TanhActivationFunctionOp : public ActivationFunctionOp<dtype> {
   public:
    TanhActivationFunctionOp(dtype* input_buffer,
                             dtype* output_buffer,
                             int size,
                             mkldnn::engine& engine)
            : ActivationFunctionOp(engine) {
        auto input_mem_index = create_memory(input_buffer, size);
        auto output_mem_index = create_memory(output_buffer, size, true);
        create_primitive(mkldnn::algorithm::eltwise_tanh,
                         input_mem_index,
                         output_mem_index);
    }
    virtual ~TanhActivationFunctionOp() {}
};

class SeluActivationFunctionOp : public ActivationFunctionOp<dtype> {
   public:
    static constexpr dtype alpha = mkl_traits<dtype>::to_type(1.6733);
    static constexpr dtype lambda = mkl_traits<dtype>::to_type(1.0507);

    SeluActivationFunctionOp(dtype* input_buffer,
                             dtype* output_buffer,
                             int size,
                             mkldnn::engine& engine)
            : ActivationFunctionOp(engine) {
        auto input_mem_index = create_memory(input_buffer, size);
        auto intermediate_mem_index = create_memory(nullptr, size);
        auto output_mem_index = create_memory(output_buffer, size);
        // SELU can be implemented using an ELU, followed by a scaling.
        create_primitive(mkldnn::algorithm::eltwise_elu,
                         input_mem_index,
                         intermediate_mem_index,
                         alpha);
        // The linear algorithm performs y = Ax + B.
        create_primitive(mkldnn::algorithm::eltwise_linear,
                         intermediate_mem_index,
                         output_mem_index,
                         lambda);
    }
    virtual ~SeluActivationFunctionOp() {}
};

class SoftmaxActivationFunctionOp : public ActivationFunctionOp<dtype> {
   public:
    SoftmaxActivationFunctionOp(dtype* input_buffer,
                                dtype* output_buffer,
                                int batch_size,
                                int softmax_size,
                                mkldnn::engine& engine)
            : ActivationFunctionOp(engine) {
        auto input_mem_index =
                create_memory(input_buffer, softmax_size, batch_size);
        auto output_mem_index =
                create_memory(output_buffer, softmax_size, batch_size, true);
        create_primitive(input_mem_index, output_mem_index);
    }

    mem_ref_t create_memory(dtype* buffer,
                            int softmax_size,
                            int batch_size,
                            bool is_output = false) {
        return BaseMklOp<dtype>::create_memory(
                buffer,
                mem_dims({ batch_size, softmax_size }),
                mem_fmt::nc,
                is_output);
    }

    virtual mkldnn::softmax_forward& create_primitive(mem_ref_t input,
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

    virtual ~SoftmaxActivationFunctionOp() {}
};

BaseMklOpPtr sigmoid(float* activations,
                     int size,
                     mkldnn::engine& cpu,
                     float* results);
BaseMklOpPtr relu(float* activations,
                  int size,
                  mkldnn::engine& cpu,
                  float* results);
BaseMklOpPtr elu(float* activations,
                 int size,
                 mkldnn::engine& cpu,
                 float* results);
BaseMklOpPtr selu(float* activations,
                  int size,
                  mkldnn::engine& cpu,
                  float* results);
BaseMklOpPtr tanh(float* activations,
                  int size,
                  mkldnn::engine& cpu,
                  float* results);
BaseMklOpPtr softmax(float* a,
                     int num_test_cases,
                     int softmax_size,
                     float* results);

void activation_fun(float* activations,
                    int batch_size,
                    int input_size,
                    activation_type function,
                    float* results,
                    device_t* device);
}  // namespace nnet_mkl

#endif
