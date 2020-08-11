#ifndef _OPERATORS_SMV_SMV_ELU_OP_H_
#define _OPERATORS_SMV_SMV_ELU_OP_H_

#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/elu_op.h"
#include "smaug/operators/smv/smv_unary_op_common.h"

namespace smaug {

/** Elementwise exponential linear unit on SMV. */
class SmvEluOp : public EluOp<SmvBackend> {
   public:
    using EluOp<SmvBackend>::EluOp;
    void tile() override { tiledTensors = smv::unary::doTiling(this, false); }
    void run() override { smv::unary::run(this, tiledTensors); };

   protected:
    std::array<TiledTensor, 2> tiledTensors;
};

/** Elementwise scaled exponential linear unit on SMV. */
class SmvSeluOp : public SeluOp<SmvBackend> {
   public:
    using SeluOp<SmvBackend>::SeluOp;
    void tile() override { tiledTensors = smv::unary::doTiling(this, false); }
    void run() override { smv::unary::run(this, tiledTensors); };

   protected:
    std::array<TiledTensor, 2> tiledTensors;
};

}  // namespace smaug

#endif

