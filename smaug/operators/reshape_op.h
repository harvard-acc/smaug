#ifndef _OPERATORS_RESHAPE_OP_H_
#define _OPERATORS_RESHAPE_OP_H_

#include "smaug/core/backend.h"
#include "smaug/core/operator.h"
#include "smaug/core/tensor_utils.h"

namespace smaug {

/** \ingroup Operators
 *
 * Implements the reshape operator, which changes the dimensionality of a
 * Tensor while retaining the same number of elements. The output need not be
 * of the same DataLayout.
 *
 * @tparam Backend The Backend specialization of this Operator.
 */
template <typename Backend>
class ReshapeOp : public Operator {
   public:
    ReshapeOp(const std::string& name, Workspace* workspace)
            : Operator(name, OpType::Reshape, workspace) {
        inputs.resize(1, nullptr);
        outputs.resize(1, nullptr);
    }

    ReshapeOp(const std::string& name,
              Workspace* workspace,
              const std::vector<int>& _shape,
              DataLayout _layout)
            : Operator(name, OpType::Reshape, workspace), shape(_shape),
              layout(_layout) {
        inputs.resize(1, nullptr);
        outputs.resize(1, nullptr);
    }

    /** Set the desired shape of the output. */
    void setShape(const std::vector<int>& _shape, DataLayout _layout) {
        shape = _shape;
        layout = _layout;
    }
    /** Set the desired shape of the output. */
    void setShape(const std::initializer_list<int>& _shape,
                  DataLayout _layout) {
        shape = _shape;
        layout = _layout;
    }

    void createAllTensors() override {
        Tensor* input = getInput(0);
        Tensor* output = new Tensor(
                name, TensorShape(shape, layout, Backend::Alignment));
        workspace->addTensor(output);
        outputs.at(0) = output;
    }

    void run() override {
        // Copy the input data.
        Tensor* input = getInput(0);
        Tensor* output = getOutput(0);
        const TensorShape& inputShape = input->getShape();
        const TensorShape& outputShape = output->getShape();
        int inputNumDims = input->ndims();
        int outputNumDims = output->ndims();
        int inputPadding = inputShape.getPadding(inputNumDims - 1);
        int outputPadding = outputShape.getPadding(outputNumDims - 1);
        if (inputPadding == outputPadding) {
            // If input and output have the same padding, then copy the raw
            // tensor data.
            // TODO(xyzsam): Must also check for DataLayout here.
            copyRawTensorData(output, input, 0, 0, inputShape.storageSize());
        } else {
            // Due to different paddings, we can't copy the data in one shot.
            copyTensorData(output,
                           input,
                           std::vector<int>(outputNumDims, 0),
                           std::vector<int>(inputNumDims, 0),
                           inputShape.size());
        }
     }

   protected:
    std::vector<int> shape;
    DataLayout layout;
};

}  // namespace smaug

#endif
