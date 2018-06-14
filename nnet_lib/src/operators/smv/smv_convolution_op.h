#ifndef _OPERATORS_SMV_SMV_CONVOLUTION_OP_H_
#define _OPERATORS_SMV_SMV_CONVOLUTION_OP_H_

#include "core/backend.h"
#include "operators/common.h"
#include "operators/convolution_op.h"

namespace smaug {

namespace smv {
namespace conv {

extern const int kNumPEs;
extern const int kNumMaccsPerPE;

using SmvTensor = Tensor<SmvBackend>;

}  // namespace conv
}  // namespace smv

class SmvConvolutionOp : public ConvolutionOp<SmvBackend> {
  public:
    using ConvolutionOp<SmvBackend>::ConvolutionOp;
    virtual void run();
};

}  // namespace smaug

#endif
