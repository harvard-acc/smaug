#ifndef _OPERATORS_SMV_SMV_ELU_OP_H_
#define _OPERATORS_SMV_SMV_ELU_OP_H_

#include "core/backend.h"
#include "operators/common.h"
#include "operators/elu_op.h"
#include "operators/smv/smv_unary_op_common.h"

namespace smaug {

class SmvEluOp : public EluOp<SmvBackend> {
   public:
    using EluOp<SmvBackend>::EluOp;
    void tile() override { tiledTensors = smv::unary::doTiling(this, false); }
    void run() override { smv::unary::run(this); };

   protected:
    std::array<TiledTensor, 2> tiledTensors;
};

class SmvSeluOp : public SeluOp<SmvBackend> {
   public:
    using SeluOp<SmvBackend>::SeluOp;
    void tile() override { tiledTensors = smv::unary::doTiling(this, false); }
    void run() override { smv::unary::run(this); };

   protected:
    std::array<TiledTensor, 2> tiledTensors;
};

}  // namespace smaug

#endif

