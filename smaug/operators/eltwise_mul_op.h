#ifndef _OPERATORS_ELTWISE_MUL_OP_H_
#define _OPERATORS_ELTWISE_MUL_OP_H_

#include "core/backend.h"
#include "core/operator.h"
#include "core/tensor_utils.h"
#include "core/workspace.h"

namespace smaug {

template <typename Backend>
class EltwiseMulOp : public Operator {
   public:
    EltwiseMulOp(const std::string& name, Workspace* workspace)
            : Operator(name, OpType::EltwiseMul, workspace) {
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
        out << this->name << " (EltwiseMul)\t\t" << outputShape << "\n";
    }

   protected:
    enum { Input0, Input1, kNumInputs };
    enum { Outputs, kNumOutputs };
};

REGISTER_SPECIAL_OP(EltwiseMulOp, ReferenceBackend);

}  // namespace smaug

#endif
