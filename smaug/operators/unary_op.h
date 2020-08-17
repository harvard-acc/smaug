#ifndef _OPERATORS_UNARY_OP_H_
#define _OPERATORS_UNARY_OP_H_

#include <string>

#include "smaug/core/operator.h"
#include "smaug/core/workspace.h"

namespace smaug {

/** \ingroup Operators
 *
 * A base class for all unary operators: operators that only take a single
 * input. Unary operators can produce multiple output Tensors.
 *
 * @tparam Backend The Backend specialization of this Operator.
 */
template <typename Backend>
class UnaryOp : public Operator {
   public:
    UnaryOp(const std::string& name, OpType opType, Workspace* workspace)
            : Operator(name, opType, workspace) {
        inputs.resize(kNumInputs, nullptr);
        outputs.resize(kNumOutputs, nullptr);
    }

    bool validate() override { return Operator::validate(); }

    void createAllTensors() override {
        createOutputTensors();
    }

    void createOutputTensors() {
        if (outputs[Outputs])
            return;
        TensorShape shape = inputs.at(Inputs)->getShape();
        Tensor* output = new Tensor(name, shape);
        workspace->addTensor(output);
        outputs[Outputs] = output;
    }

    enum { Inputs, kNumInputs };
    enum { Outputs, kNumOutputs };
};

}  // namespace smaug

#endif
