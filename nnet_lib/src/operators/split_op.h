#ifndef _OPERATORS_SPLIT_OP_H_
#define _OPERATORS_SPLIT_OP_H_

#include <vector>
#include <initializer_list>

#include "core/backend.h"
#include "core/operator.h"
#include "core/tensor_utils.h"

namespace smaug {

template <typename Backend>
class SplitOp : public Operator {
   public:
    SplitOp(const std::string& name, Workspace* workspace)
            : Operator(name, OpType::Split, workspace) {
        inputs.resize(1, nullptr);
    }

    SplitOp(const std::string& name,
            Workspace* workspace,
            const std::vector<int>& _splits,
            int axis = 0)
            : Operator(name, OpType::Split, workspace), splits(_splits),
              splitAxis(axis) {
        inputs.resize(1, nullptr);
        outputs.resize(splits.size());
    }

    void setSplits(const std::vector<int>& _splits) {
        splits = _splits;
        outputs.resize(splits.size());
    }
    void setSplits(const std::initializer_list<int>& _splits) {
        splits = _splits;
        outputs.resize(splits.size());
    }
    void setSplitAxis(int axis) { splitAxis = axis; }

    const std::vector<int>& getSplits() const { return splits; }
    int getSplitAxis() const { return splitAxis; }

    bool validate() override {
        int splitSum = 0;
        for (int i = 0; i < splits.size(); i++)
            splitSum += splits[i];
        return (splitSum == inputs.at(0)->dim(splitAxis) &&
                Operator::validate());
    }

    void createAllTensors() override {
        std::vector<int> dims = getInput(0)->getShape().dims();
        DataLayout layout = getInput(0)->getShape().getLayout();
        for (int i = 0; i < splits.size(); i++) {
            dims[splitAxis] = splits[i];
            TensorShape shape(dims, layout, Backend::Alignment);
            Tensor* output = new Tensor(name + std::to_string(i), shape);
            workspace->addTensor(output);
            outputs.at(i) = output;
        }
    }

    void run() override {
        Tensor* input = getInput(0);
        int ndims = input->ndims();
        std::vector<int> srcOrigin(ndims, 0);
        for (int i = 0; i < getOutputs().size(); i++) {
            Tensor* output = getOutput(i);
            copyTensorRegion(output,
                             input,
                             std::vector<int>(ndims, 0),
                             srcOrigin,
                             output->getShape().dims());
            srcOrigin[splitAxis] += output->dim(splitAxis);
        }
    }

    void printSummary(std::ostream& out) const override {
        out << this->name << " (Split)\t\t";
        for (int i = 0; i < outputs.size(); i++) {
            out << outputs.at(0)->getShape();
            if (i != outputs.size() - 1)
                out << ",";
            else
                out << "\n";
        }
    }

   protected:
    int splitAxis;
    std::vector<int> splits;
};

}  // namespace smaug

#endif
