#ifndef _OPERATORS_CONCAT_OP_H_
#define _OPERATORS_CONCAT_OP_H_

#include "core/backend.h"
#include "core/operator.h"
#include "core/tensor_utils.h"

namespace smaug {

template <typename Backend>
class ConcatOp : public Operator {
   public:
    ConcatOp(const std::string& name, Workspace* workspace)
            : Operator(name, OpType::Concat, workspace) {
        outputs.resize(1, nullptr);
    }

    ConcatOp(const std::string& name,
             Workspace* workspace,
             int num,
             int axis = 0)
            : Operator(name, OpType::Concat, workspace), concatAxis(axis) {
        inputs.resize(num);
        outputs.resize(1, nullptr);
    }

    void setNumInputs(int num) { inputs.resize(num); }
    void setConcatAxis(int axis) { concatAxis = axis; }

    TensorShape inferOutputShape() const {
        assert(getInputs().size() > 0 && "Unable to get inputs for concat op!");
        std::vector<int> dims = getInput(0)->getShape().dims();
        DataLayout layout = getInput(0)->getShape().getLayout();
        int dim = 0;
        for (int i = 0; i < getInputs().size(); i++) {
            dim += getInput(i)->dim(concatAxis);
        }
        dims[concatAxis] = dim;
        return TensorShape(dims, layout, Backend::Alignment);
    }

    void createOutputTensor() {
        TensorShape shape = inferOutputShape();
        Tensor* output = new Tensor(name, shape);
        workspace->addTensor(output);
        outputs.at(0) = output;
    }

    void createAllTensors() override{
        createOutputTensor();
    }

    void run() override {
        Tensor* output = getOutput(0);
        int ndims = output->ndims();
        std::vector<int> dstOrigin(ndims, 0);
        for (int i = 0; i < getInputs().size(); i++) {
            Tensor* input = getInput(i);
            copyTensorRegion(output,
                             input,
                             dstOrigin,
                             std::vector<int>(ndims, 0),
                             input->getShape().dims());
            dstOrigin[concatAxis] += input->dim(concatAxis);
        }
    }

    void printSummary(std::ostream& out) const override {
        const TensorShape& outputShape = outputs.at(0)->getShape();
        out << this->name << " (Concat)\t\t" << outputShape << "\n";
    }

    int getConcatAxis() const { return concatAxis; }

   protected:
    int concatAxis;
};

}  // namespace smaug

#endif
