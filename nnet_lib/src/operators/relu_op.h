#ifndef _OPERATORS_RELU_OP_H_
#define _OPERATORS_RELU_OP_H_

#include <string>

#include "operators/unary_op.h"

namespace smaug {

template <typename Backend>
class ReluOp : public UnaryOp<Backend> {
   public:
    ReluOp(const std::string& name, Workspace* workspace, float _slope = 0)
            : UnaryOp<Backend>(name, OpType::ReLU, workspace), slope(_slope) {}

    virtual void run() {}
    virtual std::string opTypeName() const {
        return slope == 0 ? "ReLU" : "LReLU";
    }
    void setSlope(float _slope) { slope = _slope; }

   protected:
    // Slope in the negative region.
    float slope;
};

}  // namespace smaug

#endif
