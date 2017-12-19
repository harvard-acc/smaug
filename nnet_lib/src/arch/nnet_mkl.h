#ifndef _ARCH_MKL_H_
#define _ARCH_MKL_H_

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

    const mem_d& get_output_mem_desc() const {
        return memories.at(output_idx).get_primitive_desc().desc();
    }

   protected:
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
                            bool is_output = false) {
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
            memories.emplace_back(std::make_shared<mkldnn::memory>(target_desc));
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

class MklSession {
   public:
    MklSession() : cpu(mkldnn::engine::cpu, 0) {}

    // Stream object.
    mkldnn::engine cpu;
};

}  // namespace nnet_mkl

#endif
