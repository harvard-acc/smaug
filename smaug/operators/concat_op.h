#ifndef _OPERATORS_CONCAT_OP_H_
#define _OPERATORS_CONCAT_OP_H_

#include "smaug/core/backend.h"
#include "smaug/core/operator.h"
#include "smaug/core/tensor_utils.h"

namespace smaug {

/** \ingroup Operators
 * Concatenates N Tensors along a specified axis.
 *
 * This has a software-based implementation.
 *
 * @tparam Backend The Backend that sets Alignment.
 */
template <typename Backend>
class ConcatOp : public Operator {
   public:
    ConcatOp(const std::string& name, Workspace* workspace)
            : Operator(name, OpType::Concat, workspace) {
        outputs.resize(1, nullptr);
    }

    /**
     * Create a ConcatOp.
     *
     * @param name Operator name
     * @param workspace Workspace to manage this Operator.
     * @param num Number of tensors to concatenate.
     * @param axis Axis/dimension along which to concatenate.
     */
    ConcatOp(const std::string& name,
             Workspace* workspace,
             int num,
             int axis = 0)
            : Operator(name, OpType::Concat, workspace), concatAxis(axis) {
        inputs.resize(num);
        outputs.resize(1, nullptr);
    }

    /** Set the number of Tensors to concatenate. */
    void setNumInputs(int num) { inputs.resize(num); }
    /** Set the axis along which to concatenate. */
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

    int getConcatAxis() const { return concatAxis; }

   protected:
    int concatAxis;
};

}  // namespace smaug

#endif
