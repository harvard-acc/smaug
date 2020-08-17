#ifndef _OPERATORS_GREATER_OP_H_
#define _OPERATORS_GREATER_OP_H_

#include "smaug/core/backend.h"
#include "smaug/core/operator.h"
#include "smaug/core/tensor_utils.h"
#include "smaug/core/workspace.h"
#include "smaug/operators/eltwise_op.h"

namespace smaug {

/** \ingroup Operators
 * Implements an elementwise greater than operator.
 *
 * @tparam Backend The Backend specialization of this Operator.
 */
template <typename Backend>
class GreaterOp : public EltwiseOp<Backend> {
   public:
    GreaterOp(const std::string& name, Workspace* workspace)
            : EltwiseOp<Backend>(name, OpType::Greater, workspace) {}

    void run() override {
        assert(false && "Please implement the run() method for this backend!");
    }
};

/** \ingroup Operators
 * Implements an elementwise greater than or equal to operator.
 *
 * @tparam Backend The Backend specialization of this Operator.
 */
template <typename Backend>
class GreaterEqualOp : public EltwiseOp<Backend> {
   public:
    GreaterEqualOp(const std::string& name, Workspace* workspace)
            : EltwiseOp<Backend>(name, OpType::GreaterEqual, workspace) {}

    void run() override {
        assert(false && "Please implement the run() method for this backend!");
    }
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
REGISTER_SPECIAL_OP(GreaterOp, ReferenceBackend);
REGISTER_SPECIAL_OP(GreaterEqualOp, ReferenceBackend);
#endif

}  // namespace smaug

#endif
