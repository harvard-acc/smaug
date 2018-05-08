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

    virtual void printSummary(std::ostream& out) const {
        const TensorShape& weightsShape =
                this->inputs.at(Parent::Kernels)->getShape();
        const TensorShape& outputShape =
                this->outputs.at(Parent::Outputs)->getShape();
        out << this->name << " (DepthwiseConvolution)\t\t" << outputShape
            << "\t\t" << weightsShape << "\t\t" << weightsShape.total() << "\n";
        /*
        out << "  Row, col strides: (" << this->rowStride << ", "
            << this->colStride << ")\n"; */
    }
};

}  // namespace smaug

#endif
