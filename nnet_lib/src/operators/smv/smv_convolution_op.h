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
    void run() override;
    DataLayoutSet getInputDataLayouts() const override {
        return DataLayoutSet(DataLayout::NHWC);
    }
    DataLayoutSet getOutputDataLayouts() const override {
        return DataLayoutSet(DataLayout::NHWC);
    }
    friend class smv::conv::TilingOptimizer;

  protected:
   void runNHWC(TiledTensor& inputs,
                TiledTensor& weights,
                TiledTensor& outputs);
   void invokeSystolicArrayKernel(unsigned accelId,
                                  float16* inputs,
                                  float16* weights,
                                  float16* outputs,
                                  int inputsDims[4],
                                  int weightsDims[4],
                                  int outputsDims[4],
                                  int inputsPad,
                                  int weightsPad,
                                  int outputPad,
                                  int inputHaloPad[4],
                                  int stride,
                                  int ifmapStart,
                                  int kernStart,
                                  bool accumulate,
                                  bool readInputs,
                                  bool readWeights,
                                  bool sendResults,
                                  ActivationInfo* actInfo);
};

}  // namespace smaug

#endif
