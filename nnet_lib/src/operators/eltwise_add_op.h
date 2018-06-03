#ifndef _OPERATORS_ELTWISE_ADD_OP_H_
#define _OPERATORS_ELTWISE_ADD_OP_H_

#include "core/backend.h"
#include "core/operator.h"
#include "core/workspace.h"

namespace smaug {

template <typename Backend>
class EltwiseAddOp : public Operator {
   public:
    EltwiseAddOp(const std::string& name, Workspace* workspace)
            : Operator(name, OpType::EltwiseAdd, workspace) {
        inputs.resize(kNumInputs, nullptr);
        outputs.resize(kNumOutputs, nullptr);
    }

    virtual void run() {}
    TensorShape inferOutputShape() const {
        return getInput<Backend>(Input0)->getShape();
    }
    virtual DataLayoutSet getInputDataLayouts() const {
        return DataLayoutSet(DataLayout::X);
    }
    virtual DataLayoutSet getOutputDataLayouts() const {
        return DataLayoutSet(DataLayout::X);
    }
    void createOutputTensors() {
        TensorShape shape = inferOutputShape();
        Tensor<Backend>* output = new Tensor<Backend>(name, shape);
        outputs.at(Outputs) = output;
        workspace->addTensor(output);
    }
    virtual void createAllTensors() {
        createOutputTensors();
    }
    virtual void printSummary(std::ostream& out) const {
      const TensorShape& outputShape = outputs.at(Outputs)->getShape();
      out << this->name << " (EltwiseAdd)\t\t" << outputShape << "\n";
    }

   protected:
    enum { Input0, Input1, kNumInputs };
    enum { Outputs, kNumOutputs };
};

REGISTER_SPECIAL_OP(EltwiseAddOp, ReferenceBackend);

}  // namespace smaug

#endif
