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
    void run() override { smv::unary::run(this); };
};

class SmvSeluOp : public SeluOp<SmvBackend> {
   public:
    using SeluOp<SmvBackend>::SeluOp;
    void run() override { smv::unary::run(this); };
};

}  // namespace smaug

#endif

