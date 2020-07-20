#ifndef _OPERATORS_LSTM_CELL_OP_H_
#define _OPERATORS_LSTM_CELL_OP_H_

#include <string>

#include "smaug/core/backend.h"

namespace smaug {

template <typename Backend>
class LSTMCellOp : public Operator {
   public:
    LSTMCellOp(const std::string& name, OpType opType, Workspace* workspace)
            : Operator(name, opType, workspace) {
        inputs.resize(kNumInputs, nullptr);
        outputs.resize(kNumOutputs, nullptr);
    }

    bool validate() override { return Operator::validate(); }

    void createAllTensors() override {
        // Infer the output shapes using the input shapes.
        Tensor* inputsX = getInput(InputsX);
        Tensor* InputsH =  getInput(InputsH);
        Tensor* weightsX = getInput(WeightX);
        TensorShape outputShape({ batch, outputSize },
                                input->getShape().getLayout(),
                                Backend::Alignment);
        Tensor* outputH = new Tensor(name + "_H", outputShape);
        Tensor* outputC = new Tensor(name + "_C", outputShape);
    }

    enum { InputsX, InputsH, InputsC, WeightX, WeightsH, kNumInputs };
    enum { OutputsH, OutputsC, kNumOutputs };
};

};
