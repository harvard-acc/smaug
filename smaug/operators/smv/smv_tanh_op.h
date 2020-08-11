#ifndef _OPERATORS_SMV_SMV_TANH_OP_H_
#define _OPERATORS_SMV_SMV_TANH_OP_H_

#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/tanh_op.h"
#include "smaug/operators/smv/smv_unary_op_common.h"

namespace smaug {

/** Tanh operator on SMV. */
class SmvTanhOp : public TanhOp<SmvBackend> {
   public:
    using TanhOp<SmvBackend>::TanhOp;
    void tile() override { tiledTensors = smv::unary::doTiling(this, false); }
    void run() override { smv::unary::run(this, tiledTensors); }

   protected:
    std::array<TiledTensor, 2> tiledTensors;
};

/** Hard tanh operator on SMV. */
class SmvHardTanhOp : public HardTanhOp<SmvBackend> {
   public:
    using HardTanhOp<SmvBackend>::HardTanhOp;
    void tile() override { tiledTensors = smv::unary::doTiling(this, false); }
    void run() override { smv::unary::run(this, tiledTensors); }

   protected:
    std::array<TiledTensor, 2> tiledTensors;
};

}  // namespace smaug

#endif

