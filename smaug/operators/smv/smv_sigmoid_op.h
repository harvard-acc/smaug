#ifndef _OPERATORS_SMV_SMV_SIGMOID_OP_H_
#define _OPERATORS_SMV_SMV_SIGMOID_OP_H_

#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/sigmoid_op.h"
#include "smaug/operators/smv/smv_unary_op_common.h"

namespace smaug {

/** Sigmoid linear-unit operator on SMV. */
class SmvSigmoidOp : public SigmoidOp<SmvBackend> {
   public:
    using SigmoidOp<SmvBackend>::SigmoidOp;
    void tile() override { tiledTensors = smv::unary::doTiling(this, false); }
    void run() override { smv::unary::run(this, tiledTensors); }

   protected:
    std::array<TiledTensor, 2> tiledTensors;
};

}  // namespace smaug

#endif
