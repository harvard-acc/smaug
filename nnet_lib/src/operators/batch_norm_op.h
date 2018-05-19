#ifndef _OPERATORS_BATCH_NORM_OP_H_
#define _OPERATORS_BATCH_NORM_OP_H_

#include "core/operator.h"
#include "core/tensor.h"
#include "core/workspace.h"
// TODO: This might be dangerous...
#include "operators/reorder_op.h"

namespace smaug {

template <typename Backend>
class BatchNormOp : public Operator {
   public:
    BatchNormOp(const std::string& name, Workspace* workspace)
            : Operator(name, OpType::BatchNorm, workspace),
              meanName(name + "/mean"), varianceName(name + "/variance"),
              gammaName(name + "/gamma"), betaName(name + "/beta") {
        inputs.resize(kNumInputs, nullptr);
        outputs.resize(kNumOutputs, nullptr);
    }

    virtual void run() {}
    TensorShape inferOutputShape() const {
        return getInput<Backend>(Inputs)->getShape();
    }
    TensorShape inferWeightsShape() const {
        TensorShape shape = getInput<Backend>(Inputs)->getShape();
        DataLayout layout = shape.getLayout();
        int ndims = shape.size();
        if (ndims >= 4) {
            // This is a volume which should be batch norm'ed by feature map.
            bool isNCHW = layout == DataLayout::NCHW;
            int fmaps = isNCHW ? shape[ndims - 3] : shape[ndims - 1];
            return TensorShape({ fmaps }, DataLayout::X);
        } else if (ndims == 2) {
            if (layout == DataLayout::NC)
                return TensorShape({ shape[1] }, DataLayout::X);
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
        inputs[Mean] = new Tensor<Backend>(meanName, shape);
        inputs[Variance] = new Tensor<Backend>(varianceName, shape);
        inputs[Gamma] = new Tensor<Backend>(gammaName, shape);
        inputs[Beta] = new Tensor<Backend>(betaName, shape);
        for (int i = Mean; i <= Beta; i++)
            workspace->addTensor(static_cast<Tensor<Backend>*>(inputs[i]));
    }

    void createOutputTensors() {
        if (outputs[Outputs])
            return;
        TensorShape shape = inferOutputShape();
        Tensor<Backend>* output = new Tensor<Backend>(name, shape);
        workspace->addTensor<Backend>(output);
        outputs[Outputs] = output;
    }

    virtual void createAllTensors() {
        createWeightsTensors();
        createOutputTensors();
    }

    virtual DataLayoutSet getInputDataLayouts() const {
        // We have to keep going backwards until we find the first operation
        // from which we can deduce the structure of this operation's input.
        /*
        switch (sourceOp->getOpType()) {
            case Convolution3d:
            case ConvolutionDepthwise:
            case MaxPooling:
            case AveragePooling:
                return DataLayoutSet(DataLayout::NCHW);
            case InnerProduct:
                return DataLayoutSet(DataLayout::NC);
            case Reorder:
                return dynamic_cast<ReorderOp<Backend>*>(sourceOp)
                        ->getTargetDataLayout();
            default:
                assert(false && "No valid input op for batch norm!");
        }
        */
        return DataLayoutSet(DataLayout::UnknownLayout);
    }

    virtual DataLayoutSet getOutputDataLayouts() const {
        return getInputDataLayouts();
    }

    virtual void printSummary(std::ostream& out) const {
      const TensorShape& weightsShape = inputs.at(Mean)->getShape();
      const TensorShape& outputShape = outputs.at(Outputs)->getShape();
      out << this->name << " (BatchNormalization)\t" << outputShape << "\t\t"
          << weightsShape << "\t\t\t" << 4 * weightsShape.total() << "\n";
    }

    virtual std::vector<TensorBase*> getParameterizableInputs() {
        return { inputs[Mean], inputs[Variance], inputs[Gamma], inputs[Beta] };
    }

   protected:
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

    const std::string meanName;
    const std::string varianceName;
    const std::string gammaName;
    const std::string betaName;
};

}  // namespace smaug

#endif
