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
    void run() override { smv::unary::run(this); };
};

}  // namespace smaug

#endif
