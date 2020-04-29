#ifndef _OPERATORS_SMV_SMV_RELU_OP_H_
#define _OPERATORS_SMV_SMV_RELU_OP_H_

#include "core/backend.h"
#include "operators/common.h"
#include "operators/relu_op.h"
#include "operators/smv/smv_unary_op_common.h"

namespace smaug {

class SmvReluOp : public ReluOp<SmvBackend> {
   public:
    using ReluOp<SmvBackend>::ReluOp;
    void tile() override { tiledTensors = smv::unary::doTiling(this, false); }
    void run() override { smv::unary::run(this, tiledTensors); };

   protected:
    std::array<TiledTensor, 2> tiledTensors;
};

}  // namespace smaug

#endif
