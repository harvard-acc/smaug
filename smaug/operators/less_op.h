#ifndef _OPERATORS_LESS_OP_H_
#define _OPERATORS_LESS_OP_H_

#include "smaug/core/backend.h"
#include "smaug/core/operator.h"
#include "smaug/core/tensor_utils.h"
#include "smaug/core/workspace.h"
#include "smaug/operators/eltwise_op.h"

namespace smaug {

/** \ingroup Operators
 * Implements an elementwise less-than operator.
 *
 * @tparam Backend The Backend specialization of this Operator.
 */
template <typename Backend>
class LessOp : public EltwiseOp<Backend> {
   public:
    LessOp(const std::string& name, Workspace* workspace)
            : EltwiseOp<Backend>(name, OpType::Less, workspace) {}

    void run() override {
        assert(false && "Please implement the run() method for this backend!");
    }
};

/** \ingroup Operators
 * Implements an elementwise less-than-or-equal-to operator.
 *
 * @tparam Backend The Backend specialization of this Operator.
 */
template <typename Backend>
class LessEqualOp : public EltwiseOp<Backend> {
   public:
    LessEqualOp(const std::string& name, Workspace* workspace)
            : EltwiseOp<Backend>(name, OpType::LessEqual, workspace) {}

    void run() override {
        assert(false && "Please implement the run() method for this backend!");
    }
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
REGISTER_SPECIAL_OP(LessOp, ReferenceBackend);
REGISTER_SPECIAL_OP(LessEqualOp, ReferenceBackend);
#endif

}  // namespace smaug

#endif
