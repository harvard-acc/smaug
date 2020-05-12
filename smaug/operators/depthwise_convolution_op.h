#ifndef _OPERATORS_DEPTHWISE_CONVOLUTION_OP_H_
#define _OPERATORS_DEPTHWISE_CONVOLUTION_OP_H_

#include "smaug/operators/convolution_op.h"

namespace smaug {

template <typename Backend>
class DepthwiseConvolutionOp : public ConvolutionOp<Backend> {
   protected:
    typedef ConvolutionOp<Backend> Parent;

   public:
    DepthwiseConvolutionOp(const std::string& name, Workspace* workspace)
            : ConvolutionOp<Backend>(name, workspace) {
        this->template opType = OpType::ConvolutionDepthwise;
    }

    void run() override {}

    TensorShape inferOutputShape() const override {
        Tensor* input = this->template getInput(Parent::Inputs);
        assert(input && "Unable to get input for convolution op!");
        const TensorShape& shape = input->getShape();
        DataLayout layout = shape.getLayout();
        bool isNCHW = (layout == DataLayout::NCHW);
        int rowIdx = isNCHW ? 2 : 1;
        int colIdx = isNCHW ? 3 : 2;
        int outputRows = this->computeOutputDim(shape[rowIdx],
                                                this->weightRows,
                                                this->rowStride,
                                                this->paddingType);
        int outputCols = this->computeOutputDim(shape[colIdx],
                                                this->weightCols,
                                                this->colStride,
                                                this->paddingType);
        if (isNCHW) {
            return TensorShape({ shape[0], shape[1], outputRows, outputCols },
                               layout,
                               Backend::Alignment);
        } else {
            return TensorShape({ shape[0], outputRows, outputCols, shape[3] },
                               layout,
                               Backend::Alignment);
        }
    }

    TensorShape inferWeightsShape() const override {
        Tensor* input = this->template getInput(Parent::Inputs);
        const TensorShape& shape = input->getShape();
        DataLayout layout = shape.getLayout();
        bool isNCHW = (layout == DataLayout::NCHW);
        int inputChannels = isNCHW ? shape[1] : shape[3];
        if (isNCHW) {
            return TensorShape(
                    { 1, inputChannels, this->weightRows, this->weightCols },
                    layout, Backend::Alignment);
        } else {
            return TensorShape(
                    { 1, this->weightRows, this->weightCols, inputChannels },
                    layout, Backend::Alignment);
        }
    }

    void printSummary(std::ostream& out) const override {
        const TensorShape& weightsShape =
                this->inputs.at(Parent::Kernels)->getShape();
        const TensorShape& outputShape =
                this->outputs.at(Parent::Outputs)->getShape();
        out << this->name << " (DepthwiseConvolution)\t\t" << outputShape
            << "\t\t" << weightsShape << "\t\t" << weightsShape.size() << "\n";
        /*
        out << "  Row, col strides: (" << this->rowStride << ", "
            << this->colStride << ")\n"; */
    }
};

REGISTER_SPECIAL_OP(DepthwiseConvolutionOp, ReferenceBackend);

}  // namespace smaug

#endif
