#ifndef _OPERATORS_CONVOLUTION_OP_H_
#define _OPERATORS_CONVOLUTION_OP_H_

#include <string>

#include "core/backend.h"
#include "core/operator.h"
#include "core/workspace.h"
#include "core/tensor_utils.h"
#include "core/types.pb.h"
#include "operators/common.h"
#include "operators/fused_activation_op.h"

namespace smaug {

template <typename Backend>
class ConvolutionOp : public FusedActivationOp {
   public:
    ConvolutionOp(const std::string& name, Workspace* workspace)
            : FusedActivationOp(name, OpType::Convolution3d, workspace),
              weightRows(0), weightCols(0), numOfmaps(0), rowStride(0),
              colStride(0), paddingType(UnknownPadding), inputPadding(4),
              weightsName(name + "/kernels"), sampling({ NoSampling, 1 }) {
        inputs.resize(kNumInputs, nullptr);
        outputs.resize(kNumOutputs, nullptr);
    }

    void setWeightDims(int _weightRows, int _weightCols, int _numOfmaps) {
        weightRows = _weightRows;
        weightCols = _weightCols;
        numOfmaps = _numOfmaps;
    }

    void setStride(int _rowStride, int _colStride) {
        rowStride = _rowStride;
        colStride = _colStride;
    }

    void setPadding(PaddingType padding) {
        paddingType = padding;
    }

    bool validate() override {
        return (weightRows > 0 && weightCols > 0 && numOfmaps > 0 &&
                rowStride > 0 && colStride > 0 &&
                paddingType != UnknownPadding && Operator::validate());
    }

    virtual TensorShape inferOutputShape() const {
        Tensor* input = getInput(Inputs);
        assert(input && "Unable to get input for convolution op!");
        DataLayout layout = input->getShape().getLayout();
        bool isNCHW = (layout == DataLayout::NCHW);
        int rowIdx = isNCHW ? 2 : 1;
        int colIdx = isNCHW ? 3 : 2;
        int outputRows = computeOutputDim(
                input->dim(rowIdx), weightRows, rowStride, paddingType);
        int outputCols = computeOutputDim(
                input->dim(colIdx), weightCols, colStride, paddingType);
        if (isNCHW) {
            return TensorShape({ 1, numOfmaps, outputRows, outputCols },
                               layout,
                               Backend::Alignment);
        } else {
            return TensorShape({ 1, outputRows, outputCols, numOfmaps },
                               layout,
                               Backend::Alignment);
        }
    }

    virtual TensorShape inferWeightsShape() const {
        Tensor* input = getInput(Inputs);
        DataLayout layout = input->getShape().getLayout();
        bool isNCHW = (layout == DataLayout::NCHW);
        int channelsIdx = isNCHW ? 1 : 3;
        int inputChannels = input->dim(channelsIdx);
        if (isNCHW) {
            return TensorShape(
                    { numOfmaps, inputChannels, weightRows, weightCols },
                    layout, Backend::Alignment);
        } else {
            return TensorShape(
                    { numOfmaps, weightRows, weightCols, inputChannels },
                    layout, Backend::Alignment);
        }
    }

    // Create placeholder tensors for weights, assuming that any data layout is
    // okay. This function can be specialized for a specific backend.
    void createWeightsTensors() {
        if (inputs.at(Kernels) != nullptr)
            return;
        TensorShape shape = inferWeightsShape();
        Tensor* kernels = new Tensor(weightsName, shape);
        workspace->addTensor(kernels);
        inputs[Kernels] = kernels;
    }

    void createOutputTensors() {
        if (outputs.at(Outputs) != nullptr)
            return;
        TensorShape shape = inferOutputShape();
        Tensor* output = new Tensor(name, shape);
        workspace->addTensor(output);
        outputs.at(Outputs) = output;
    }

    void createAllTensors() override {
        createWeightsTensors();
        createOutputTensors();
        computeInputPadding();
    }

    int getNumOfmaps() const { return numOfmaps; }

    void run() override {}

    void printSummary(std::ostream& out) const override {
      const TensorShape& weightsShape = inputs.at(Kernels)->getShape();
      const TensorShape& outputShape = outputs.at(Outputs)->getShape();
      out << this->name << " (Convolution3d)\t\t" << outputShape << "\t\t"
          << weightsShape << "\t\t" << weightsShape.size() << "\n";
    }

    std::vector<TensorBase*> getParameterizableInputs() override {
        return { inputs[Kernels] };
    }

    int getRowStride() const { return rowStride; }
    int getColStride() const { return colStride; }
    int getWeightRows() const { return weightRows; }
    int getWeightCols() const { return weightCols; }
    PaddingType getPadding() const { return paddingType; }
    const std::vector<int>& getInputPadding() const { return inputPadding; }

    bool isSamplingSupported() const override { return true; }
    void setSamplingInfo(const SamplingInfo& _sampling) override {
        sampling = _sampling;
    }

   protected:
    int computeOutputDim(int inputDim,
                         int weightDim,
                         int stride,
                         PaddingType pad) const {
        int padding = (pad == SamePadding ? (weightDim - 1) : 0);
        return computeOutputDim(inputDim, weightDim, stride, padding);
    }
    int computeOutputDim(int inputDim,
                         int weightDim,
                         int stride,
                         int padding) const {
        return (inputDim - weightDim + padding) / stride + 1;
    }

    // We precompute the input halos here.
    void computeInputPadding() {
        int totalRowPad = (paddingType == SamePadding) ? weightRows - 1 : 0;
        int totalColPad = (paddingType == SamePadding) ? weightCols - 1 : 0;
        inputPadding[0] = FRAC_CEIL(totalRowPad, 2);
        inputPadding[1] = totalRowPad - inputPadding[0];
        inputPadding[2] = FRAC_CEIL(totalColPad, 2);
        inputPadding[3] = totalColPad - inputPadding[2];
    }

  public:
    enum { Inputs, Kernels, kNumInputs };
    enum { Outputs, kNumOutputs };

  protected:
    int weightRows;
    int weightCols;
    int numOfmaps;
    int rowStride;
    int colStride;
    PaddingType paddingType;
    // Padding sizes on the four boundaries of the input 2D feature map.
    std::vector<int> inputPadding;
    std::string weightsName;
    SamplingInfo sampling;
};

REGISTER_SPECIAL_OP(ConvolutionOp, ReferenceBackend);

}  // namespace smaug

#endif
