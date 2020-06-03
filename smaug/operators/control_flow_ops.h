#ifndef _OPERATORS_CONTROL_FLOW_OPS_H_
#define _OPERATORS_CONTROL_FLOW_OPS_H_

#include "smaug/core/backend.h"
#include "smaug/core/operator.h"
#include "smaug/core/tensor_utils.h"

namespace smaug {

template <typename Backend>
class SwitchOp : public Operator {
   public:
    enum { Input, Pred, kNumInputs };
    enum { OutputFalse, OutputTrue, kNumOutputs };

    SwitchOp(const std::string& name, Workspace* workspace)
            : Operator(name, OpType::Switch, workspace) {
        inputs.resize(2, nullptr);
        outputs.resize(2, nullptr);
    }

    bool validate() override {
        if (getInput(Pred)->getShape().size() != 1)
            return false;
        return Operator::validate();
    }

    void createAllTensors() override {
        Tensor* input = getInput(Input);
        TensorShape shape = inputs.at(Input)->getShape();
        Tensor* outputFalse = new Tensor(name + "_false", shape);
        Tensor* outputTrue = new Tensor(name + "_true", shape);
        workspace->addTensor(outputFalse);
        workspace->addTensor(outputTrue);
        outputs.at(OutputFalse) = outputFalse;
        outputs.at(OutputTrue) = outputTrue;
    }

    void run() override {
        Tensor* input = getInput(Input);
        Tensor* outputFalse = getOutput(OutputFalse);
        Tensor* outputTrue = getOutput(OutputTrue);
        const TensorShape& inputShape = input->getShape();
        Tensor* predTensor = getInput(Pred);
        bool* pred = predTensor->data<bool>();
        if (pred[0]) {
            outputFalse->setDead();
            copyRawTensorData(
                    outputTrue, input, 0, 0, inputShape.storageSize());
        } else {
            outputTrue->setDead();
            copyRawTensorData(
                    outputFalse, input, 0, 0, inputShape.storageSize());
        }
    }
};

}  // namespace smaug

#endif
