#ifndef _OPERATORS_UNARY_OP_H_
#define _OPERATORS_UNARY_OP_H_

#include <string>

#include "smaug/core/operator.h"
#include "smaug/core/tensor_utils.h"
#include "smaug/core/workspace.h"

namespace smaug {

template <typename Backend>
class UnaryOp : public Operator {
   public:
    UnaryOp(const std::string& name, OpType opType, Workspace* workspace)
            : Operator(name, opType, workspace) {
        inputs.resize(kNumInputs, nullptr);
        outputs.resize(kNumOutputs, nullptr);
    }

    bool validate() override { return Operator::validate(); }
    virtual std::string opTypeName() const = 0;

    void createAllTensors() override {
        createOutputTensors();
    }
    void printSummary(std::ostream& out) const override {
        TensorShape outputShape = outputs.at(Outputs)->getShape();
        out << this->name << " (" << opTypeName() << ")\t\t" << outputShape
            << "\n";
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
