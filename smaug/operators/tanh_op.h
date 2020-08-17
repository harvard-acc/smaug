#ifndef _OPERATORS_TANH_OP_H_
#define _OPERATORS_TANH_OP_H_

#include <string>

#include "smaug/core/backend.h"
#include "smaug/operators/unary_op.h"

namespace smaug {

/** \ingroup Operators
 * Implements the tanh operator.
 *
 * @tparam Backend The Backend specialization of this Operator.
 */
template <typename Backend>
class TanhOp : public UnaryOp<Backend> {
   public:
    TanhOp(const std::string& name, Workspace* workspace)
            : UnaryOp<Backend>(name, OpType::Tanh, workspace) {}

    void run() override {}
};

/** \ingroup Operators
 * Implements the hard tanh operator, which bounds the min and max value of the
 * tanh operator.
 *
 * @tparam Backend The Backend specialization of this Operator.
 */
template <typename Backend>
class HardTanhOp : public UnaryOp<Backend> {
   public:
    HardTanhOp(const std::string& name,
               Workspace* workspace,
               float _min = -1,
               float _max = 1)
            : UnaryOp<Backend>(name, OpType::HardTanh, workspace), min(_min),
              max(_max) {}

    void run() override {}

    void setMin(float _min) { min = _min; }
    void setMax(float _max) { max = _max; }
    float getMin() const { return min; }
    float getMax() const { return max; }

   protected:
    float min;
    float max;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
REGISTER_SPECIAL_OP(TanhOp, ReferenceBackend);
REGISTER_SPECIAL_OP(HardTanhOp, ReferenceBackend);
#endif

}  // namespace smaug

#endif
