#ifndef _OPERATORS_ELTWISE_ADD_OP_H_
#define _OPERATORS_ELTWISE_ADD_OP_H_

#include "smaug/core/backend.h"
#include "smaug/core/operator.h"
#include "smaug/core/tensor_utils.h"
#include "smaug/core/workspace.h"

namespace smaug {

template <typename Backend>
class EltwiseAddOp : public Operator {
   public:
    EltwiseAddOp(const std::string& name, Workspace* workspace)
            : Operator(name, OpType::EltwiseAdd, workspace) {
        inputs.resize(kNumInputs, nullptr);
        outputs.resize(kNumOutputs, nullptr);
    }

    void run() override {}
    TensorShape inferOutputShape() const {
        return getInput(Input0)->getShape();
    }
    void createOutputTensors() {
        TensorShape shape = inferOutputShape();
        Tensor* output = new Tensor(name, shape);
        outputs.at(Outputs) = output;
        workspace->addTensor(output);
    }
    void createAllTensors() override { createOutputTensors(); }
    void printSummary(std::ostream& out) const override {
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
