#ifndef _ARCH_MKL_H_
#define _ARCH_MKL_H_

#include <memory>

#include "mkldnn.hpp"

#include "core/nnet_fwd_defs.h"

namespace nnet_mkl {

using mem_d = mkldnn::memory::desc;
using mem_pd = mkldnn::memory::primitive_desc;

class MklSession {
   public:
    MklSession() : cpu(mkldnn::engine::cpu, 0) {}

    // Stream object.
    mkldnn::engine cpu;
};

}  // namespace nnet_mkl

#endif
