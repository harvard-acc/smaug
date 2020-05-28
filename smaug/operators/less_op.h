#ifndef _OPERATORS_LESS_OP_H_
#define _OPERATORS_LESS_OP_H_

#include "smaug/core/backend.h"
#include "smaug/core/operator.h"
#include "smaug/core/tensor_utils.h"
#include "smaug/core/workspace.h"
#include "smaug/operators/eltwise_op.h"

namespace smaug {

template <typename Backend>
class LessOp : public EltwiseOp<Backend> {
   public:
    LessOp(const std::string& name, Workspace* workspace)
            : EltwiseOp<Backend>(name, OpType::Less, workspace) {}

    void run() override {
        assert(false && "Please implement the run() method for this backend!");
    }
};

template <typename Backend>
class LessEqualOp : public EltwiseOp<Backend> {
   public:
    LessEqualOp(const std::string& name, Workspace* workspace)
            : EltwiseOp<Backend>(name, OpType::LessEqual, workspace) {}

    void run() override {
        assert(false && "Please implement the run() method for this backend!");
    }
};

REGISTER_SPECIAL_OP(LessOp, ReferenceBackend);
REGISTER_SPECIAL_OP(LessEqualOp, ReferenceBackend);

}  // namespace smaug

#endif
