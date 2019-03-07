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

class TilingOptimizer;

}  // namespace conv
}  // namespace smv

class SmvConvolutionOp : public ConvolutionOp<SmvBackend> {
  public:
    using ConvolutionOp<SmvBackend>::ConvolutionOp;
    virtual void run();
    virtual DataLayoutSet getInputDataLayouts() const {
        return DataLayoutSet(DataLayout::NHWC);
    }
    virtual DataLayoutSet getOutputDataLayouts() const {
        return DataLayoutSet(DataLayout::NHWC);
    }
    friend class smv::conv::TilingOptimizer;

  protected:
   void runNHWC(TiledTensor& inputs,
                TiledTensor& weights,
                TiledTensor& outputs);
};

}  // namespace smaug

#endif
