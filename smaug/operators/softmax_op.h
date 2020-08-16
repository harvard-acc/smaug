#ifndef _OPERATORS_SOFTMAX_OP_H_
#define _OPERATORS_SOFTMAX_OP_H_

#include <string>

#include "smaug/core/backend.h"
#include "smaug/operators/unary_op.h"

namespace smaug {

/** \ingroup Operators
 *
 * Implements the softmax operator.
 *
 * @tparam Backend The Backend specialization of this Operator.
 */
template <typename Backend>
class SoftmaxOp : public UnaryOp<Backend> {
   public:
    SoftmaxOp(const std::string& name, Workspace* workspace)
            : UnaryOp<Backend>(name, OpType::Softmax, workspace) {}

    void run() override {}
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
REGISTER_SPECIAL_OP(SoftmaxOp, ReferenceBackend);
#endif

}  // namespace smaug

#endif
