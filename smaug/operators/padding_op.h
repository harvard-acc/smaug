#ifndef _OPERATORS_PADDING_OP_H_
#define _OPERATORS_PADDING_OP_H_

#include "smaug/core/backend.h"
#include "smaug/core/operator.h"
#include "smaug/core/tensor.h"
#include "smaug/core/workspace.h"
#include <google/protobuf/repeated_field.h>
using namespace google::protobuf;

namespace smaug {

/** \ingroup Operators
 * \brief Pad a given tensor in  any number of dimensions with arbitrary size.
 *
 * This has a software-based implementation.
 *
 * @tparam Backend The Backend that sets Alignment.
 */
template <typename Backend>
class PaddingOp : public Operator {
   public:
    PaddingOp(const std::string& name, Workspace* workspace)
            : Operator(name, OpType::Padding, workspace) {
        inputs.resize(kNumInputs, nullptr);
        outputs.resize(kNumOutputs, nullptr);
    }

    /**
     * Set the paddingSize of the Tensor along each dimension.
     * The paddingSize is orgainized as <dim1_forward, dim1_backward, ...
     * ,dimk_backward>
     */
    void setPaddingSize(RepeatedField<google::protobuf::int32> const& val) {
        std::vector<int> paddingSize(val.begin(), val.end());
    }

    void setPaddingSize(std::vector<int> const& val) { paddingSize = val; }

    const std::vector<int> getPaddingSize() { return paddingSize; }

    void run() override {
        Tensor* input = getInput(0);
        Tensor* output = getOutput(0);
        int ndims = input->ndims();
        const std::vector<int> inputDims = input->getShape().dims();
        const std::vector<int> outputDims = output->getShape().dims();
        int total_dim = 1;
        for (int i : outputDims) {
            total_dim *= i;
        }
        std::vector<float> vf(total_dim, 0);
        output->fillData(vf.data(), vf.size());
        std::vector<int> destOrigin, paddingBegin, srcOrigin;
        for (int i = 0; i < ndims; i++) {
            paddingBegin.push_back(paddingSize[2 * i]);
            srcOrigin.push_back(0);
        }
        destOrigin = std::vector<int>(paddingBegin);
        std::vector<int> regionSize = inputDims;
        copyTensorRegion(output, input, destOrigin, srcOrigin, regionSize);
    }

    // Optional override for testing purposes.
    void createAllTensors() override {
        Tensor* input = getInput(0);
        int ndims = input->ndims();
        std::vector<int> dims = input->getShape().dims();
        for (int i = 0; i < ndims; i++) {
            dims[i] += (paddingSize[2 * i] + paddingSize[2 * i + 1]);
        }
        TensorShape shape(
                dims, input->getShape().getLayout(), Backend::Alignment);
        Tensor* output = new Tensor(name, shape);
        workspace->addTensor(output);
        outputs.at(0) = output;
    }

    // Optional but recommended function to verify operator parameters.
    bool validate() override {
        Tensor* input = getInput(0);
        int ndims = input->ndims();
        if (paddingSize.size() != 2 * ndims) {
            return false;
        }
        return Operator::validate();
    }

    enum { kInputs, kNumInputs };
    enum { kOutputs, kNumOutputs };

   private:
    std::vector<int> paddingSize = {};
};

}  // namespace smaug

#endif