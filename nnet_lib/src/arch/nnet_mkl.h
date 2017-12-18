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

class MklSession {
   public:
    MklSession() : cpu(mkldnn::engine::cpu, 0) {}

    // Stream object.
    mkldnn::engine cpu;
};

}  // namespace nnet_mkl

#endif
