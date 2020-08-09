#ifndef _OPERATORS_FUSED_ACTIVATION_OP_H_
#define _OPERATORS_FUSED_ACTIVATION_OP_H_

#include <string>

#include "smaug/core/operator.h"
#include "smaug/core/tensor_utils.h"
#include "smaug/core/workspace.h"

namespace smaug {

/** \ingroup Operators
 * An Operator fused with an activation function.
 *
 * This is an optimized operator that reduces memory/compute by directly
 * computing the activation function on its output.  This is a parent class of
 * all operator implementations that support activation op fusion.
 */
class FusedActivationOp : public Operator {
   public:
    FusedActivationOp(const std::string& name,
                      OpType opType,
                      Workspace* workspace)
            : Operator(name, opType, workspace) {}

    void setActivation(ActivationInfo _actInfo) { actInfo = _actInfo; }

    ActivationInfo getActivation() const { return actInfo; }

   protected:
    ActivationInfo actInfo;
};

}  // namespace smaug

#endif
