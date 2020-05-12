#ifndef _OPERATORS_FUSED_ACTIVATION_OP_H_
#define _OPERATORS_FUSED_ACTIVATION_OP_H_

#include <string>

#include "smaug/core/operator.h"
#include "smaug/core/tensor_utils.h"
#include "smaug/core/workspace.h"

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
