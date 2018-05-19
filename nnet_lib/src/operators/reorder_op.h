#ifndef _CORE_REORDER_OP_H_
#define _CORE_REORDER_OP_H_

#include "core/operator.h"

namespace smaug {

template <typename Backend>
class ReorderOp : public Operator {
   public:
    ReorderOp(const std::string& name,
              DataLayout _targetLayout,
              Workspace* workspace)
            : Operator(name, OpType::Reorder, workspace),
              targetLayout(_targetLayout) {
        inputs.resize(kNumInputs, nullptr);
        outputs.resize(kNumOutputs, nullptr);
    }

    ReorderOp(const std::string& name, Workspace* workspace)
            : ReorderOp(name,
                        DataLayout::UnknownLayout,
                        OpType::Reorder,
                        workspace) {}

    DataLayout getTargetDataLayout() const { return targetLayout; }
    void setTargetLayout(DataLayout layout) { targetLayout = layout; }

    virtual void run() {}

    virtual bool validate() {
        if (!Operator::validate())
            return false;
        DataLayout sourceLayout = inputs[Inputs]->getShape().getLayout();
        if (sourceLayout == DataLayout::UnknownLayout) {
            std::cerr << "[ERROR]: Reorder operation has unknown source "
                         "layout!\n";
            return false;
        }
        if (targetLayout == DataLayout::UnknownLayout) {
            std::cerr << "[ERROR]: Reorder operation has unknown target "
                         "layout!\n";
            return false;
        }
        if (sourceLayout == targetLayout) {
            std::cerr << "[ERROR]: Reorder operation does not change the data "
                         "layout!\n";
            return false;
        }
        return true;
    }

    TensorShape inferOutputShape() const {
        TensorShape inputShape = getInput<Backend>(Inputs)->getShape();
        std::vector<int> dims(2, 1);
        dims[0] = inputShape[0];
        for (int i = 1; i < inputShape.size(); ++i) {
            dims[1] *= inputShape[i];
        }
        return TensorShape(dims, targetLayout);
    }

    void createOutputTensors() {
        assert(targetLayout != DataLayout::UnknownLayout &&
               "Cannot create output tensor with unknown target data layout!");
        TensorShape shape = inferOutputShape();
        Tensor<Backend>* output = new Tensor<Backend>(name, shape);
        workspace->addTensor(output);
        outputs.at(Outputs) = output;
    }
    virtual void createAllTensors() {
        createOutputTensors();
    }
    virtual DataLayoutSet getInputDataLayouts() const {
        // TODO: Use the input tensor.
        return DataLayoutSet(DataLayout::UnknownLayout);
    }
    virtual DataLayoutSet getOutputDataLayouts() const {
        return DataLayoutSet(targetLayout);
    }
    virtual void printSummary(std::ostream& out) const {
        const TensorShape& shape = outputs.at(Outputs)->getShape();
        out << name << " (Reorder)\t\t" << shape << "\n";
    }

   protected:
    enum { Inputs, kNumInputs };
    enum { Outputs, kNumOutputs };
    DataLayout targetLayout;
};

template <typename Backend>
class FlattenOp : public ReorderOp<Backend> {
   public:
    typedef ReorderOp<Backend> Parent;

    FlattenOp(const std::string& name, Workspace* workspace)
            : ReorderOp<Backend>(name, DataLayout::NC, workspace) {}

    virtual void printSummary(std::ostream& out) const {
        const TensorShape& shape =
                this->outputs.at(Parent::Outputs)->getShape();
        out << this->name << " (Flatten)\t\t" << shape << "\n";
    }
};

}  // namespace smaug

#endif
