#ifndef _OPERATORS_SIGMOID_OP_H_
#define _OPERATORS_SIGMOID_OP_H_

#include <string>

#include "smaug/core/backend.h"
#include "smaug/operators/unary_op.h"

namespace smaug {

/** \ingroup Operators
 *
 * \brief Implements the sigmoid operator, defined as 1/(1 + exp(-input)).
 *
 * @tparam Backend The Backend specialization of this Operator.
 */
template <typename Backend>
class SigmoidOp : public UnaryOp<Backend> {
   public:
    SigmoidOp(const std::string& name, Workspace* workspace)
            : UnaryOp<Backend>(name, OpType::Sigmoid, workspace) {}

    void run() override {}
};

REGISTER_SPECIAL_OP(SigmoidOp, ReferenceBackend);

}  // namespace smaug

#endif
