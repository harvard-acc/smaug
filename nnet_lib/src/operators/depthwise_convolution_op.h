#ifndef _OPERATORS_DEPTHWISE_CONVOLUTION_OP_H_
#define _OPERATORS_DEPTHWISE_CONVOLUTION_OP_H_

#include "operators/convolution_op.h"

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

    virtual void run() {}

    virtual TensorShape inferOutputShape() const {
        Tensor<Backend>* input =
                this->template getInput<Backend>(Parent::Inputs);
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
            return TensorShape(
                    { shape[0], shape[1], outputRows, outputCols }, layout);
        } else {
            return TensorShape(
                    { shape[0], outputRows, outputCols, shape[3] }, layout);
        }
    }

    virtual TensorShape inferWeightsShape() const {
        Tensor<Backend>* input =
                this->template getInput<Backend>(Parent::Inputs);
        const TensorShape& shape = input->getShape();
        DataLayout layout = shape.getLayout();
        bool isNCHW = (layout == DataLayout::NCHW);
        int inputChannels = isNCHW ? shape[1] : shape[3];
        int padding = calc_padding(
                isNCHW ? this->weightCols : inputChannels, Backend::Alignment);
        if (isNCHW) {
            return TensorShape({ 1, inputChannels, this->weightRows,
                                 this->weightCols + padding },
                               layout);
        } else {
            return TensorShape({ 1, this->weightRows, this->weightCols,
                                 inputChannels + padding },
                               layout);
        }
    }

    virtual void printSummary(std::ostream& out) const {
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
