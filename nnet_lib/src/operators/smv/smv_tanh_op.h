#ifndef _OPERATORS_SMV_SMV_TANH_OP_H_
#define _OPERATORS_SMV_SMV_TANH_OP_H_

#include "core/backend.h"
#include "operators/common.h"
#include "operators/tanh_op.h"
#include "operators/smv/smv_unary_op_common.h"

namespace smaug {

class SmvTanhOp : public TanhOp<SmvBackend> {
   public:
    using TanhOp<SmvBackend>::TanhOp;
    virtual void run() { smv::unary::run(this); };
};

class SmvHardTanhOp : public HardTanhOp<SmvBackend> {
   public:
    using HardTanhOp<SmvBackend>::HardTanhOp;
    virtual void run() { smv::unary::run(this); };
};

}  // namespace smaug

#endif

