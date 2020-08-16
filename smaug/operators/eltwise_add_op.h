#ifndef _OPERATORS_ELTWISE_ADD_OP_H_
#define _OPERATORS_ELTWISE_ADD_OP_H_

#include "smaug/core/backend.h"
#include "smaug/core/operator.h"
#include "smaug/core/tensor_utils.h"
#include "smaug/core/workspace.h"
#include "smaug/operators/eltwise_op.h"

namespace smaug {

/** \ingroup Operators
 * Adds two Tensors elementwise.
 *
 * @tparam Backend The Backend specialization of this Operator.
 */
template <typename Backend>
class EltwiseAddOp : public EltwiseOp<Backend> {
   public:
    EltwiseAddOp(const std::string& name, Workspace* workspace)
            : EltwiseOp<Backend>(name, OpType::EltwiseAdd, workspace) {}

    void run() override {}
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
REGISTER_SPECIAL_OP(EltwiseAddOp, ReferenceBackend);
#endif

}  // namespace smaug

#endif
