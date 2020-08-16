#ifndef _OPERATORS_CONTROL_FLOW_OPS_H_
#define _OPERATORS_CONTROL_FLOW_OPS_H_

#include "smaug/core/backend.h"
#include "smaug/core/operator.h"
#include "smaug/core/tensor_utils.h"

namespace smaug {

/** \ingroup Operators
 * The switch operator passes an input Tensor to one of two output tensors,
 * depending on whether the specified predicate is true. The other tensor is
 * marked as dead.
 *
 * This is an integral component of implementing control flow in networks.
 *
 * @tparam Backend The Backend specialization of this Operator.
 */
template <typename Backend>
class SwitchOp : public Operator {
   public:
    enum {
      /** The input Tensor to pass through. */
      Input,
      /** A scalar Tensor (a Tensor with just one value). 0 is false; anything
       * nonzero is true. */
      Pred,
      kNumInputs
    };
    enum {
      /** The output tensor on the false branch. */
      OutputFalse,
      /** The output tensor on the true branch. */
      OutputTrue,
      kNumOutputs
    };

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

/** \ingroup Operators
 * A merge operator takes multiple tensors, all but one of which should be
 * dead, and copies the one live Tensor to its output.
 */
template <typename Backend>
class MergeOp : public Operator {
   public:
    MergeOp(const std::string& name, Workspace* workspace)
            : Operator(name, OpType::Merge, workspace) {
        outputs.resize(1, nullptr);
    }

    void setNumInputs(int num) { inputs.resize(num); }

    void createAllTensors() override {
        Tensor* output =
                workspace->addTensor(new Tensor(name, getInput(0)->getShape()));
        outputs.at(0) = output;
    }

    /** A merge operator is dead only when all its inputs are dead. */
    bool isDead() override {
        for (auto input : inputs) {
            if (!input->isDead())
                return false;
        }
        return true;
    }

    void run() override {
        Tensor* output = getOutput(0);
        bool forwarded = false;
        for (int i = 0; i < getInputs().size(); i++) {
            Tensor* input = getInput(i);
            if (!input->isDead()) {
                copyRawTensorData(
                        output, input, 0, 0, input->getShape().storageSize());
                forwarded = true;
                break;
            }
        }
        if (!forwarded) {
            std::cerr << "All inputs to the merge operator are dead!\n";
            exit(1);
        }
    }
};

}  // namespace smaug

#endif
