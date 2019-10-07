#ifndef _OPERATORS_SIGMOID_OP_H_
#define _OPERATORS_SIGMOID_OP_H_

#include <string>

#include "core/backend.h"
#include "operators/unary_op.h"

namespace smaug {

template <typename Backend>
class SigmoidOp : public UnaryOp<Backend> {
   public:
    SigmoidOp(const std::string& name, Workspace* workspace)
            : UnaryOp<Backend>(name, OpType::Sigmoid, workspace) {}

    void run() override {}
    std::string opTypeName() const override { return "Sigmoid"; }
};

REGISTER_SPECIAL_OP(SigmoidOp, ReferenceBackend);

}  // namespace smaug

#endif
