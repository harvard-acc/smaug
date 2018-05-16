#ifndef _OPERATORS_SIGMOID_OP_H_
#define _OPERATORS_SIGMOID_OP_H_

#include <string>

#include "operators/unary_op.h"

namespace smaug {

template <typename Backend>
class SigmoidOp : public UnaryOp<Backend> {
   public:
    SigmoidOp(const std::string& name, Workspace* workspace)
            : UnaryOp<Backend>(name, OpType::Sigmoid, workspace) {}

    virtual void run() {}
    virtual std::string opTypeName() const { return "Sigmoid"; }
};

}  // namespace smaug

#endif
