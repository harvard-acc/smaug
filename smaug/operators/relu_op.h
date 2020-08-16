#ifndef _OPERATORS_RELU_OP_H_
#define _OPERATORS_RELU_OP_H_

#include <string>

#include "smaug/core/backend.h"
#include "smaug/operators/unary_op.h"

namespace smaug {

/** \ingroup Operators
  * Implements the rectified linear unit operator: max(slope * x, 0).
 *
 * @tparam Backend The Backend specialization of this Operator.
  */
template <typename Backend>
class ReluOp : public UnaryOp<Backend> {
   public:
    ReluOp(const std::string& name, Workspace* workspace, float _slope = 0)
            : UnaryOp<Backend>(name, OpType::ReLU, workspace), slope(_slope) {}

    void run() override {}
    void setSlope(float _slope) { slope = _slope; }
    float getSlope () const { return slope; }

   protected:
    // Slope in the negative region.
    float slope;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
REGISTER_SPECIAL_OP(ReluOp, ReferenceBackend);
#endif


}  // namespace smaug

#endif
