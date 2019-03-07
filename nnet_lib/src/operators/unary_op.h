#ifndef _OPERATORS_UNARY_OP_H_
#define _OPERATORS_UNARY_OP_H_

#include <string>

#include "core/operator.h"
#include "core/workspace.h"

namespace smaug {

template <typename Backend>
class UnaryOp : public Operator {
   public:
    UnaryOp(const std::string& name, OpType opType, Workspace* workspace)
            : Operator(name, opType, workspace) {
        inputs.resize(kNumInputs, nullptr);
        outputs.resize(kNumOutputs, nullptr);
    }

    virtual void run() = 0;
    virtual bool validate() { return Operator::validate(); }
    virtual std::string opTypeName() const = 0;

    virtual DataLayoutSet getInputDataLayouts() const {
        return DataLayoutSet(DataLayout::X);
    }
    virtual DataLayoutSet getOutputDataLayouts() const {
        return DataLayoutSet(DataLayout::X);
    }
    virtual void createAllTensors() {
        createOutputTensors();
    }
    virtual void printSummary(std::ostream& out) const {
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

   protected:
    enum { Inputs, kNumInputs };
    enum { Outputs, kNumOutputs };
};

}  // namespace smaug

#endif
