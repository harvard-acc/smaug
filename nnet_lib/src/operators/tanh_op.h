#ifndef _OPERATORS_TANH_OP_H_
#define _OPERATORS_TANH_OP_H_

#include <string>

#include "core/backend.h"
#include "operators/unary_op.h"

namespace smaug {

template <typename Backend>
class TanhOp : public UnaryOp<Backend> {
   public:
    TanhOp(const std::string& name, Workspace* workspace)
            : UnaryOp<Backend>(name, OpType::Tanh, workspace) {}

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
            : UnaryOp<Backend>(name, OpType::HardTanh, workspace), min(_min),
              max(_max) {}

    virtual void run() {}
    virtual std::string opTypeName() const { return "HardTanh"; }

    void setMin(float _min) { min = _min; }
    void setMax(float _max) { max = _max; }
    float getMin() const { return min; }
    float getMax() const { return max; }

   protected:
    float min;
    float max;
};

REGISTER_SPECIAL_OP(TanhOp, ReferenceBackend);
REGISTER_SPECIAL_OP(HardTanhOp, ReferenceBackend);

}  // namespace smaug

#endif
