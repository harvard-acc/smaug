#ifndef _OPERATORS_SPLIT_OP_H_
#define _OPERATORS_SPLIT_OP_H_

#include <vector>
#include <initializer_list>

#include "smaug/core/backend.h"
#include "smaug/core/operator.h"
#include "smaug/core/tensor_utils.h"

namespace smaug {

/** \ingroup Operators
 *
 * Implements the split operator, which divides a Tensor into N output Tensors
 * along a specified dimension.
 *
 * @tparam Backend The Backend specialization of this Operator.
 */
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

    /** Set the size (along the split axis) of each split Tensor. */
    void setSplits(const std::vector<int>& _splits) {
        splits = _splits;
        outputs.resize(splits.size());
    }
    void setSplits(const std::initializer_list<int>& _splits) {
        splits = _splits;
        outputs.resize(splits.size());
    }

    /** Set the axis along which to split the input Tensor. */
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

   protected:
    int splitAxis;
    std::vector<int> splits;
};

}  // namespace smaug

#endif
