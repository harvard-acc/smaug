#ifndef _OPERATORS_SOFTMAX_OP_H_
#define _OPERATORS_SOFTMAX_OP_H_

#include <string>

#include "smaug/core/backend.h"
#include "smaug/operators/unary_op.h"

namespace smaug {

template <typename Backend>
class SoftmaxOp : public UnaryOp<Backend> {
   public:
    SoftmaxOp(const std::string& name, Workspace* workspace)
            : UnaryOp<Backend>(name, OpType::Softmax, workspace) {}

    void run() override {}
    std::string opTypeName() const override { return "Softmax"; }
};

REGISTER_SPECIAL_OP(SoftmaxOp, ReferenceBackend);

}  // namespace smaug

#endif
