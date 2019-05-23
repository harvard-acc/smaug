#ifndef _OPERATORS_BATCH_NORM_OP_H_
#define _OPERATORS_BATCH_NORM_OP_H_

#include "core/backend.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "core/tensor_utils.h"
#include "core/workspace.h"
#include "operators/common.h"
#include "operators/fused_activation_op.h"

namespace smaug {

template <typename Backend>
class BatchNormOp : public FusedActivationOp {
   public:
    enum {
        Inputs,
        Mean,
        Variance,
        Gamma,
        Scaleshift = Gamma,  // for MKL.
        Beta,
        kNumInputs
    };
    enum { Outputs, kNumOutputs };
    static constexpr float kEpsilon = 1e-5;

    BatchNormOp(const std::string& name, Workspace* workspace)
            : FusedActivationOp(name, OpType::BatchNorm, workspace),
              meanName(name + "/mean"), varianceName(name + "/variance"),
              gammaName(name + "/gamma"), betaName(name + "/beta") {
        inputs.resize(kNumInputs, nullptr);
        outputs.resize(kNumOutputs, nullptr);
    }

    virtual void run() {}
    TensorShape inferOutputShape() const {
        return getInput(Inputs)->getShape();
    }
    TensorShape inferWeightsShape() const {
        TensorShape shape = getInput(Inputs)->getShape();
        DataLayout layout = shape.getLayout();
        int ndims = shape.ndims();
        if (ndims >= 4) {
            // This is a volume which should be batch norm'ed by feature map.
            bool isNCHW = layout == DataLayout::NCHW;
            int fmaps = isNCHW ? shape[ndims - 3] : shape[ndims - 1];
            return TensorShape(
                    { 1, fmaps }, DataLayout::NC, Backend::Alignment);
        } else if (ndims == 2) {
            if (layout == DataLayout::NC)
                return TensorShape(
                        { 1, shape[1] }, DataLayout::NC, Backend::Alignment);
            else
                assert(false && "Unexpected data layout for batch norm!");
        } else {
            assert(false && "Unexpected input dimensions for batch norm!");
        }
        return TensorShape();
    }

    void createWeightsTensors() {
        if (inputs[Mean] && inputs[Variance] && inputs[Gamma] && inputs[Beta])
            return;
        TensorShape shape = inferWeightsShape();
        inputs[Mean] = new Tensor(meanName, shape);
        inputs[Variance] = new Tensor(varianceName, shape);
        inputs[Gamma] = new Tensor(gammaName, shape);
        inputs[Beta] = new Tensor(betaName, shape);
        for (int i = Mean; i <= Beta; i++)
            workspace->addTensor(static_cast<Tensor*>(inputs[i]));
    }

    void createOutputTensors() {
        if (outputs[Outputs])
            return;
        TensorShape shape = inferOutputShape();
        Tensor* output = new Tensor(name, shape);
        workspace->addTensor(output);
        outputs[Outputs] = output;
    }

    virtual void createAllTensors() {
        createWeightsTensors();
        createOutputTensors();
    }

    virtual DataLayoutSet getInputDataLayouts() const {
        return DataLayoutSet(DataLayout::UnknownLayout);
    }

    virtual DataLayoutSet getOutputDataLayouts() const {
        return DataLayoutSet(DataLayout::UnknownLayout);
    }

    virtual void printSummary(std::ostream& out) const {
      const TensorShape& weightsShape = inputs.at(Mean)->getShape();
      const TensorShape& outputShape = outputs.at(Outputs)->getShape();
      out << this->name << " (BatchNormalization)\t" << outputShape << "\t\t"
          << weightsShape << "\t\t\t" << 4 * weightsShape.size() << "\n";
    }

    virtual std::vector<TensorBase*> getParameterizableInputs() {
        return { inputs[Mean], inputs[Variance], inputs[Gamma], inputs[Beta] };
    }

   protected:
    const std::string meanName;
    const std::string varianceName;
    const std::string gammaName;
    const std::string betaName;
};

REGISTER_SPECIAL_OP(BatchNormOp, ReferenceBackend);

}  // namespace smaug

#endif
