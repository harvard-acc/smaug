#ifndef _OPERATORS_BATCH_NORM_OP_H_
#define _OPERATORS_BATCH_NORM_OP_H_

#include "smaug/core/backend.h"
#include "smaug/core/operator.h"
#include "smaug/core/tensor.h"
#include "smaug/core/workspace.h"
#include "smaug/operators/common.h"
#include "smaug/operators/fused_activation_op.h"

namespace smaug {

/** \ingroup Operators
 *
 * Implements the batch normalization layer.
 *
 * The four different parameter types to BN layers (mean, v ariance, gamma, and
 * beta) are to be provided as separate Tensors. For performance reasons, the
 * variance parameter is often precomputed as 1/sqrt(variance + eps), so
 * hardware does not need to do a square root/division operation.
 *
 * @tparam Backend The Backend specialization of this Operator.
 */
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
              gammaName(name + "/gamma"), betaName(name + "/beta"),
              sampling({ NoSampling, 1 }) {
        inputs.resize(kNumInputs, nullptr);
        outputs.resize(kNumOutputs, nullptr);
    }

    void run() override {}
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

    void createAllTensors() override {
        createWeightsTensors();
        createOutputTensors();
    }

    int getNumParameters() const override {
        return kNumInputs * inputs.at(Mean)->getShape().size();
    }

    std::vector<TensorBase*> getParameterizableInputs() override {
        return { inputs[Mean], inputs[Variance], inputs[Gamma], inputs[Beta] };
    }

    bool isSamplingSupported() const override { return true; }
    void setSamplingInfo(const SamplingInfo& _sampling) override {
        sampling = _sampling;
    }

   protected:
    const std::string meanName;
    const std::string varianceName;
    const std::string gammaName;
    const std::string betaName;
    SamplingInfo sampling;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
REGISTER_SPECIAL_OP(BatchNormOp, ReferenceBackend);
#endif

}  // namespace smaug

#endif
