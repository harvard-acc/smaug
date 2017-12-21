#ifndef _ARCH_MKL_H_
#define _ARCH_MKL_H_

#include <iostream>
#include <memory>

#include "mkldnn.hpp"

#include "core/nnet_fwd_defs.h"

namespace nnet_mkl {

using mem_d = mkldnn::memory::desc;
using mem_pd = mkldnn::memory::primitive_desc;
using mem_dtype = mkldnn::memory::data_type;
using mem_fmt = mkldnn::memory::format;
using mem_dims = mkldnn::memory::dims;
typedef const mkldnn::memory& mem_ref_t;

// This is the operational datatype used throughout the MKL backend.
using dtype = float;

// This defines an mkldnn data format type, a scaling factor if the type is a
// fixed precision type, and a conversion function from float to this type.
template <typename DType>
struct mkl_traits {};

template <>
struct mkl_traits<float> {
    static const mem_dtype dtype = mem_dtype::f32;
    // When working with fixed point types, a scaling factor is often needed to
    // increase precision.
    static constexpr float scaling_factor = 1.0;
    static constexpr float to_type(float value) { return value; }
};

// The base class for all MKL backend operations in SMAUG.
template <typename DType>
class BaseMklOp {
   public:
    BaseMklOp(mkldnn::engine& eng) : engine(eng), output_idx(-1) {}
    virtual ~BaseMklOp() {}

    void run() {
        mkldnn::stream(mkldnn::stream::kind::eager).submit(worklist).wait();
    }

    // Return the list of primitives.
    //
    // This is useful if execution should be delayed.
    std::vector<mkldnn::primitive>& get_worklist() { return worklist; }

    mem_d get_output_mem_desc() const {
        return memories.at(output_idx).get_primitive_desc().desc();
    }

    const mkldnn::primitive& get_final_primitive() const {
        return worklist.back();
    }

   protected:

    // Create a operation primitive object, handling output reformatting.
    //
    // The primitive type is determined by the template parameter primitive_t.
    // This assumes that one of the constructors of primitive_t takes as the final
    // argument a memory primitive representing the result of the output.
    //
    // Args:
    //    pd: A primitive descriptor for this primitive.
    //    output_memory: The user-managed memory to store the output.
    //    input_args: All the rest constructor arguments for primitive_t,
    //      (other than the primitive descriptor and output memory).
    //
    //  Example:
    //    create_primitive_with_output_reorder<mkldnn::convolution_forward>(
    //        conv_pd, output_memory, inputs, weights, biases);
    //
    //    will construct the convolution_forward primitive like so:
    //      mkldnn::convolution_forward(conv_pd, inputs, weights, biases,
    //                                  output_memory);
    template <typename primitive_t, typename... Args>
    void create_primitive_with_output_reorder(
            const typename primitive_t::primitive_desc& pd,
            mem_ref_t output_memory,
            Args... input_args) {
        mkldnn::memory temp_output = output_memory;
        bool reordered = false;
        if (needs_reorder(output_memory, pd.dst_primitive_desc())) {
            temp_output = create_memory(pd.dst_primitive_desc());
            reordered = true;
        }

        // Using temp_output is okay because it's just a wrapper around a shared
        // ptr?
        worklist.emplace_back(primitive_t(pd, input_args..., temp_output));

        if (reordered) {
            worklist.emplace_back(mkldnn::reorder(temp_output, output_memory));
        }
    }

    template <typename primitive_t, typename... Args>
    void create_primitive_no_output_reorder(
            const typename primitive_t::primitive_desc& pd,
            DType* output_buffer,
            Args... input_args) {
        mem_ref_t output =
                create_memory(pd.dst_primitive_desc(), output_buffer, true);
        worklist.emplace_back(primitive_t(pd, input_args..., output));
    }

    // Create a memory primitive from an existing buffer.
    //
    // A reference to the created memory primitive is returned.
    //
    // If @buffer is not NULL, then the memory primitive will use that pointer;
    // otherwise, it will allocate its own memory.
    mem_ref_t create_memory(DType* buffer,
                            mem_dims dims,
                            mem_fmt fmt,
                            bool is_output = false) {
        auto md = mem_d({ dims }, mkl_traits<DType>::dtype, fmt);
        auto mempd = mem_pd(md, engine);
        if (buffer)
            memories.emplace_back(mempd, buffer);
        else
            memories.emplace_back(mempd);
        if (is_output)
            output_idx = memories.size() - 1;
        return memories.back();
    }

    // Create a memory primitive from a primitive descriptor.
    //
    // An index to the created memory primitive is returned.
    mem_ref_t create_memory(mkldnn::memory::primitive_desc pd,
                            DType* buffer = nullptr,
                            bool is_output = false) {
        if (buffer)
            memories.emplace_back(pd, buffer);
        else
            memories.emplace_back(pd);
        if (is_output)
            output_idx = memories.size() - 1;
        return memories.back();
    }

    // Returns true if the given memory format != target descriptor format.
    bool needs_reorder(mem_ref_t current_mem,
                       const mkldnn::memory::primitive_desc& target_desc) {
        return (mem_pd(target_desc) != current_mem.get_primitive_desc());
    }

    bool needs_reorder(const mem_pd& current_mem_pd, const mem_pd& target_desc) {
        return (mem_pd(target_desc) != current_mem_pd);
    }

    // Adds a reorder primitive to the worklist if required for this memory.
    //
    // The given memory's primitive descriptor is compared against the target
    // descriptor. If they differ, then a new memory primitive to store the
    // result of the reorder will be added, and a reorder primitive will be
    // added to the worklist.
    //
    // This is meant to be used for inputs to an operation primitive, because
    // the reorder is added immediately.
    mem_ref_t reorder_input_if_needed(
            mem_ref_t current_mem,
            const mkldnn::memory::primitive_desc& target_desc) {
        if (needs_reorder(current_mem, target_desc)) {
            memories.emplace_back(target_desc);
            worklist.emplace_back(
                    mkldnn::reorder(current_mem, memories.back()));
            return memories.back();
        }
        return current_mem;
    }

    // The execution engine.
    const mkldnn::engine& engine;

    // A list of all the memory objects required. These cannot be destroyed
    // before the list of primitives has been executed!
    std::vector<mkldnn::memory> memories;

    // A list of primitives to execute.
    std::vector<mkldnn::primitive> worklist;

   private:
    // The index of the memory storing the output.
    size_t output_idx;
};

using BaseMklOpPtr = std::unique_ptr<BaseMklOp<dtype>>;

class MklSession {
   public:
    MklSession() : cpu(mkldnn::engine::cpu, 0) {}

    void run() {
        // Now actually run the network.
        for (auto& op : oplist) {
            op->run();
        }
    }

    bool empty() const { return oplist.empty(); }

    const BaseMklOpPtr& last_op() {
        assert(!empty());
        return oplist.back();
    }

    // Stream object.
    mkldnn::engine cpu;

    // List of operations to run.
    std::vector<BaseMklOpPtr> oplist;
};

// Return the session pointer in this device.
MklSession* get_session(device_t* device);

}  // namespace nnet_mkl

#endif
