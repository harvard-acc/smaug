#ifndef _OPERATORS_PADDING_OP_H_
#define _OPERATORS_PADDING_OP_H_

#include "smaug/core/backend.h"
#include "smaug/core/operator.h"
#include "smaug/core/tensor.h"
// #include "smaug/core/tensor_utils.h"
#include "smaug/core/workspace.h"

namespace smaug {

/** \ingroup Operators
 * \brief Pad a given tensor in different dimension.
 *
 * This has a software-based implementation.
 *
 * @tparam Backend The Backend that sets Alignment.
 */
template <typename Backend>
class PaddingOp : public Operator {
   public:
    PaddingOp(const std::string& name,
              Workspace* workspace)
            : Operator(name, OpType::Repeat, workspace){
        inputs.resize(kNumInputs, nullptr);
        outputs.resize(kNumOutputs, nullptr);
    }

    PaddingOp(const std::string& name,
              Workspace* workspace,
              int val)
            : Operator(name, OpType::Repeat, workspace), padder(val){
        inputs.resize(kNumInputs, nullptr);
        outputs.resize(kNumOutputs, nullptr);
    }

    /** Set the number of padders of the Tensor along each dimension. */
    void setPadder(const int& val) {
        padder = val;
        // set output size?
    }

    auto getPadder() { return padder; }

    void run() override {
      Tensor* input = getInput(0);
      Tensor* output = getOutput(0);
      int ndims = input->ndims();
      std::vector<int> inputDims = input->getShape().dims();
      std::vector<int> outputDims = output->getShape().dims();
      int total_dim = 1;
      for (int i: outputDims){
        total_dim *= i;
      }
      std::vector<float> vf(total_dim, 0);
      output->fillData(vf.data(), vf.size());
      /*
      copyTensorRegion(Tensor* dest,
                      Tensor* src,
                      const std::vector<int>& destOrigin,
                      const std::vector<int>& srcOrigin,
                      const std::vector<int>& regionSize
      */
      std::vector<int> destOrigin;
      if (input->getShape().getLayout() == DataLayout::NCHW){
        destOrigin = std::vector<int>({0, 0, padder, padder});
      }
      else if(input->getShape().getLayout() == DataLayout::NHWC){
        destOrigin = std::vector<int>({0, padder, padder, 0});
      }
      else{
        assert(false && "Invalid padding data type!");
      }
      std::vector<int> srcOrigin = std::vector<int>({0, 0, 0, 0});
      std::vector<int> regionSize = inputDims;
      copyTensorRegion(output, input, destOrigin, srcOrigin, regionSize);
    }

    // Optional override for testing purposes.
    void createAllTensors() override {
        Tensor* input = getInput(0);
        std::vector<int> dims = input->getShape().dims();
        if (input->getShape().getLayout() == DataLayout::NCHW){
          dims[2] += 2*padder;
          dims[3] += 2*padder;
        }
        else if (input->getShape().getLayout() == DataLayout::NHWC){
          dims[1] += 2*padder;
          dims[2] += 2*padder;
        }
        TensorShape shape(
                dims, input->getShape().getLayout(), Backend::Alignment);
        Tensor* output = new Tensor(name, shape);
        workspace->addTensor(output);
        outputs.at(0) = output;
     }

    // Optional but recommended function to verify operator parameters.
    bool validate() override {
      if (padder < 0){
        return false;
      }
        return Operator::validate();
    }
    
    enum { kInputs, kNumInputs };
    enum { kOutputs, kNumOutputs };

  private:
    int padder = 0;
};

}  // namespace smaug

#endif