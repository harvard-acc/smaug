#ifndef _OPERATORS_ELTWISEOP_OPS_H_
#define _OPERATORS_ELTWISEOP_OPS_H_

#include "smaug/core/backend.h"
#include "smaug/core/operator.h"
#include "smaug/core/workspace.h"

namespace smaug {

/** \ingroup Operators
 * The base class of all elementwise operators.
 *
 * @tparam Backend The Backend specialization of this Operator.
 */
template <typename Backend>
class EltwiseOp : public Operator {
   public:
    EltwiseOp(const std::string& name, OpType opType, Workspace* workspace)
            : Operator(name, opType, workspace) {
        inputs.resize(kNumInputs, nullptr);
        outputs.resize(kNumOutputs, nullptr);
    }

    void createAllTensors() override {
        Tensor* output = new Tensor(name, getInput(Input0)->getShape());
        outputs.at(Outputs) = output;
        workspace->addTensor(output);
    }

   protected:
    enum { Input0, Input1, kNumInputs };
    enum { Outputs, kNumOutputs };
};

}  // namespace smaug

#endif
