#ifndef _OPERATORS_CONVOLUTION_OP_H_
#define _OPERATORS_CONVOLUTION_OP_H_

#include <string>

#include "core/operator.h"

namespace smaug {

template <typename Backend>
class ConvolutionOp : public Operator {
   public:
    ConvolutionOp(const std::string& name, Workspace* workspace)
            : Operator(name, OpType::Convolution3d, workspace), weightRows(0),
              weightCols(0), numOfmaps(0), rowStride(0), colStride(0),
              paddingType(UnknownPadding), weightsName(name + "/kernels") {
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

    void setPadding(std::string padding) {
        if (padding == "SAME")
            paddingType = SamePadding;
        else if (padding == "VALID")
            paddingType = ValidPadding;
        else
            assert(false && "Invalid padding type!");
    }

    void setPadding(PaddingType padding) {
        paddingType = padding;
    }

    virtual bool validate() {
        return (weightRows > 0 && weightCols > 0 && numOfmaps > 0 &&
                rowStride > 0 && colStride > 0 &&
                paddingType != UnknownPadding && Operator::validate());
    }

    TensorShape inferOutputShape() const {
        Tensor<Backend>* input = getInput<Backend>(Inputs);
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
            return TensorShape(
                    { 1, numOfmaps, outputRows, outputCols }, layout);
        } else {
            return TensorShape(
                    { 1, outputRows, outputCols, numOfmaps }, layout);
        }
    }

    TensorShape inferWeightsShape() const {
        Tensor<Backend>* input = getInput<Backend>(Inputs);
        DataLayout layout = input->getShape().getLayout();
        bool isNCHW = (layout == DataLayout::NCHW);
        int channelsIdx = isNCHW ? 1 : 3;
        int inputChannels = input->dim(channelsIdx);
        int padding = calc_padding(
                isNCHW ? weightCols : inputChannels, Backend::Alignment);
        if (isNCHW) {
            return TensorShape({ numOfmaps, inputChannels, weightRows,
                                 weightCols + padding },
                               layout);
        } else {
            return TensorShape({ numOfmaps, weightRows, weightCols,
                                 inputChannels + padding },
                               layout);
        }
    }

    // Create placeholder tensors for weights, assuming that any data layout is
    // okay. This function can be specialized for a specific backend.
    void createWeightsTensors() {
        if (inputs.at(Kernels) != nullptr)
            return;
        TensorShape shape = inferWeightsShape();
        Tensor<Backend>* kernels = new Tensor<Backend>(weightsName, shape);
        workspace->addTensor(kernels);
        inputs[Kernels] = kernels;
    }

    void createOutputTensors() {
        if (outputs.at(Outputs) != nullptr)
            return;
        TensorShape shape = inferOutputShape();
        Tensor<Backend>* output = new Tensor<Backend>(name, shape);
        workspace->addTensor(output);
        outputs.at(Outputs) = output;
    }

    virtual void createAllTensors() {
        createWeightsTensors();
        createOutputTensors();
    }

    int getNumOfmaps() const { return numOfmaps; }

    virtual DataLayoutSet getInputDataLayouts() const {
        return DataLayoutSet(DataLayout::NCHW);
    }
    virtual DataLayoutSet getOutputDataLayouts() const {
        return DataLayoutSet(DataLayout::NCHW);
    }

    virtual void run() {}

    virtual void printSummary(std::ostream& out) const {
      const TensorShape& weightsShape = inputs.at(Kernels)->getShape();
      const TensorShape& outputShape = outputs.at(Outputs)->getShape();
      out << this->name << " (Convolution3d)\t\t" << outputShape << "\t\t"
          << weightsShape << "\t\t" << weightsShape.total() << "\n";
    }

    virtual std::vector<TensorBase*> getParameterizableInputs() {
        return { inputs[Kernels] };
    }

   protected:
    int computeOutputDim(int inputDim,
                         int weightDim,
                         int stride,
                         PaddingType pad) const {
        int padding = (pad == SamePadding ? (weightDim - 1) : 0);
        return (inputDim - weightDim + padding) / stride + 1;
    }

    enum { Inputs, Kernels, kNumInputs };
    enum { Outputs, kNumOutputs };

    int weightRows;
    int weightCols;
    int numOfmaps;
    int rowStride;
    int colStride;
    PaddingType paddingType;
    std::string weightsName;
};

}  // namespace smaug

#endif
