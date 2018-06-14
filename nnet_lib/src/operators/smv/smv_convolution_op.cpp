#include "core/backend.h"
#include "operators/common.h"
#include "operators/smv/smv_convolution_op.h"
#include "utility/debug_stream.h"

namespace smaug {
namespace smv {
namespace conv {

const int kNumPEs = 8;
const int kNumMaccsPerPE = 32;

}  // namespace conv
}  // namespace smv

void SmvConvolutionOp::run() {
    auto input = getInput<SmvBackend>(Inputs);
    auto kernels = getInput<SmvBackend>(Kernels);
    auto output = getOutput<SmvBackend>(Outputs);
    const TensorShape& inputShape = input->getShape();
    const TensorShape& kernelShape = kernels->getShape();
    const TensorShape& outputShape = output->getShape();
    assert(inputShape.getLayout() == DataLayout::NHWC);
    assert(kernelShape.getLayout() == DataLayout::NHWC);
    assert(outputShape.getLayout() == DataLayout::NHWC);
    dout(2) << *kernels << "\n";
}

}  // namespace smaug
