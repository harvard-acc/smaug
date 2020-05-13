#ifndef _OPERATORS_FUSED_ACTIVATION_OP_H_
#define _OPERATORS_FUSED_ACTIVATION_OP_H_

#include <string>

#include "core/operator.h"
#include "core/tensor_utils.h"
#include "core/workspace.h"

namespace smaug {

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
