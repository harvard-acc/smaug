#ifndef _MKL_ACTIVATION_FUNCTIONS_H_
#define _MKL_ACTIVATION_FUNCTIONS_H_

#include "mkldnn.hpp"

#include "arch/nnet_mkl.h"

namespace nnet_mkl {

template <typename DType>
class ActivationFunctionOp {
   public:
    using mem_index_t = int;

    ActivationFunctionOp(mkldnn::engine& eng) : engine(eng), output_idx(-1) {}
    virtual ~ActivationFunctionOp() {}

    // Execute the activation function.
    void run() {
        mkldnn::stream(mkldnn::stream::kind::eager).submit(worklist).wait();
    }

    // Return the list of primitives.
    //
    // This is useful if execution should be delayed.
    std::vector<mkldnn::primitive>& get_worklist() { return worklist; }

    // Return the memory primitive containing the output of this operation.
    mkldnn::memory& get_output_memory() { return memories.at(output_idx); }

   protected:
    // Create and return an index to a memory primitive.
    //
    // If @buffer is not NULL, then the memory primitive will use that pointer;
    // otherwise, it will allocate its own memory.
    //
    // The memory primitive is stored in a list inside this object. The index
    // corresponds to the location of the primitive in this list. It is
    // guaranteed to be stable during the lifetime of this Op object.
    virtual mem_index_t create_memory(DType* buffer,
                                      int size,
                                      bool is_output = false) {
        mkldnn::memory::dims dims = { size };
        auto md = mem_d({ dims }, mkl_traits<DType>::dtype, mem_fmt::x);
        auto mempd = mem_pd(md, engine);
        if (buffer)
            memories.emplace_back(mempd, buffer);
        else
            memories.emplace_back(mempd);
        mem_index_t retval = memories.size() - 1;
        if (is_output)
            output_idx = retval;
        return retval;
    }

    // Create a return a reference to the operation primitive.
    virtual mkldnn::eltwise_forward& create_primitive(mkldnn::algorithm alg,
                                                      mem_index_t input_idx,
                                                      mem_index_t output_idx,
                                                      DType alpha = 0,
                                                      DType beta = 0) {
        auto& input = memories.at(input_idx);
        auto& output = memories.at(output_idx);
        auto desc =
                mkldnn::eltwise_forward::desc(mkldnn::prop_kind::forward,
                                              alg,
                                              input.get_primitive_desc().desc(),
                                              alpha,
                                              beta);
        auto pd = mkldnn::eltwise_forward::primitive_desc(desc, engine);
        worklist.emplace_back(mkldnn::eltwise_forward(pd, input, output));
        return static_cast<mkldnn::eltwise_forward&>(worklist.back());
    }

   protected:
    // The execution engine.
    const mkldnn::engine& engine;

    // A list of all the memory objects required. These cannot be destroyed
    // before the list of primitives has been executed!
    std::vector<mkldnn::memory> memories;

    // A list of primitives to execute.
    std::vector<mkldnn::primitive> worklist;

    // The index of the memory storing the output.
    //
    // Subsequent operations can use this to directly get a reference to the
    // memory primitive, instead of creating a new one.
    mem_index_t output_idx;
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

void sigmoid(float* activations, int size, mkldnn::engine& cpu, float* results);
void relu(float* activations, int size, mkldnn::engine& cpu, float* results);
void elu(float* activations, int size, mkldnn::engine& cpu, float* results);
void selu(float* activations, int size, mkldnn::engine& cpu, float* results);
void tanh(float* activations, int size, mkldnn::engine& cpu, float* results);

void activation_fun(float* activations,
                    int size,
                    activation_type function,
                    float* results,
                    device_t* device);
}  // namespace nnet_mkl

#endif
