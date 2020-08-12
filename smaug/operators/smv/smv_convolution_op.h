#ifndef _OPERATORS_SMV_SMV_CONVOLUTION_OP_H_
#define _OPERATORS_SMV_SMV_CONVOLUTION_OP_H_

#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/convolution_op.h"

namespace smaug {

namespace smv {
/** Contains convolution implementations and tiling optimizers for SMV. */
namespace conv {

extern const int kNumPEs;
extern const int kNumMaccsPerPE;

class TilingOptimizer;

}  // namespace conv
}  // namespace smv

/**
 * SMV backend implementation of convolution.
 *
 * The dataflow is inspired by the NVDLA convolution engine.
 */
class SmvConvolutionOp : public ConvolutionOp<SmvBackend> {
  public:
    using ConvolutionOp<SmvBackend>::ConvolutionOp;
    void tile() override;
    void run() override;
    friend class smv::conv::TilingOptimizer;

  protected:
   void runNHWC(TiledTensor& inputs,
                TiledTensor& weights,
                TiledTensor& outputs);
   std::unique_ptr<volatile int> invokeSystolicArrayKernel(
           unsigned accelId,
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

   std::array<TiledTensor, 3> tiledTensors;
};

}  // namespace smaug

#endif
