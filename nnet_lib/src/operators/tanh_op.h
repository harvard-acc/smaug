#ifndef _OPERATORS_TANH_OP_H_
#define _OPERATORS_TANH_OP_H_

#include <string>

#include "operators/unary_op.h"

namespace smaug {

template <typename Backend>
class TanhOp : public UnaryOp<Backend> {
   public:
    TanhOp(const std::string& name, Workspace* workspace)
            : UnaryOp<Backend>(name, OpType::ReLU, workspace) {}

    virtual void run() {}
    virtual std::string opTypeName() const { return "Tanh"; }
};

template <typename Backend>
class HardTanhOp : public UnaryOp<Backend> {
   public:
    HardTanhOp(const std::string& name,
               Workspace* workspace,
               float _min = -1,
               float _max = 1)
            : UnaryOp<Backend>(name, OpType::ReLU, workspace), min(_min),
              max(_max) {}

    virtual void run() {}
    virtual std::string opTypeName() const { return "HardTanh"; }

    void setMin(float _min) { min = _min; }
    void setMax(float _max) { max = _max; }

   protected:
    float min;
    float max;
};

}  // namespace smaug

#endif
