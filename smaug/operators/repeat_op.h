#ifndef _OPERATORS_REPEAT_OP_H_
#define _OPERATORS_REPEAT_OP_H_

#include <vector>
#include <initializer_list>

#include "smaug/core/backend.h"
#include "smaug/core/operator.h"
#include "smaug/core/tensor_utils.h"

namespace smaug {

/** \ingroup Operators
 *
 * Implements a repeat operator, which replicates the contents of a Tensor
 * along each dimension a configurable number of times. This is set by the
 * `setMultiples` function.
 *
 * @tparam Backend The Backend specialization of this Operator.
 */
template <typename Backend>
class RepeatOp : public Operator {
   public:
    RepeatOp(const std::string& name, Workspace* workspace)
            : Operator(name, OpType::Repeat, workspace) {
        inputs.resize(1, nullptr);
        outputs.resize(1, nullptr);
    }

    RepeatOp(const std::string& name,
             Workspace* workspace,
             const std::vector<int> _multiples)
            : Operator(name, OpType::Repeat, workspace), multiples(_multiples) {
        inputs.resize(1, nullptr);
        outputs.resize(1, nullptr);
    }

    /** Set the number of copies of the Tensor along each dimension. */
    void setMultiples(const std::vector<int>& _multiples) {
        multiples = _multiples;
    }

    /** Set the number of copies of the Tensor along each dimension. */
    void setMultiples(const std::initializer_list<int>& _multiples) {
        multiples = _multiples;
    }

    bool validate() override {
        for (int multiple : multiples) {
            if (multiple == 0)
                return false;
        }
        return Operator::validate();
    }

    void createAllTensors() override {
        Tensor* input = getInput(0);
        std::vector<int> dims = input->getShape().dims();
        for (int i = 0; i < multiples.size(); i++)
            dims[i] *= multiples[i];
        TensorShape shape(
                dims, input->getShape().getLayout(), Backend::Alignment);
        Tensor* output = new Tensor(name, shape);
        workspace->addTensor(output);
        outputs.at(0) = output;
    }

    void run() override {
        Tensor* input = getInput(0);
        Tensor* output = getOutput(0);
        int ndims = input->ndims();
        std::vector<int> inputDims = input->getShape().dims();
        std::vector<int> outputDims = output->getShape().dims();
        std::vector<int> srcOrigin = std::vector<int>(ndims, 0);
        // Copy the first piece of input into output.
        copyTensorRegion(output, input, srcOrigin, srcOrigin, inputDims);
        for (int i = ndims - 1; i >= 0; i--) {
            std::vector<int> currCopyRegion = inputDims;
            for (int j = i + 1; j < ndims; j++)
                currCopyRegion[j] = outputDims[j];
            std::vector<int> dstOrigin(ndims, 0);
            dstOrigin[i] = inputDims[i];
            while (dstOrigin[i] + currCopyRegion[i] <= outputDims[i]) {
                copyTensorRegion(
                        output, output, dstOrigin, srcOrigin, currCopyRegion);
                dstOrigin[i] += currCopyRegion[i];
                // Double the copy size for the next iteration.
                currCopyRegion[i] *= 2;
            }
            // Copy the remaining part if there's any.
            if (dstOrigin[i] < outputDims[i]) {
                currCopyRegion[i] = outputDims[i] - dstOrigin[i];
                copyTensorRegion(
                        output, output, dstOrigin, srcOrigin, currCopyRegion);
            }
        }
    }

   protected:
    std::vector<int> multiples;
};

}  // namespace smaug

#endif
