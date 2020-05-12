#ifndef _OPERATORS_SMV_SMV_BATCH_NORM_OP_H_
#define _OPERATORS_SMV_SMV_BATCH_NORM_OP_H_

#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/batch_norm_op.h"

namespace smaug {

namespace smv {
namespace bn {

extern const int kVectorSize;

class TilingOptimizer;

}  // namespace bn
}  // namespace smv

class SmvBatchNormOp : public BatchNormOp<SmvBackend> {
  public:
    using BatchNormOp<SmvBackend>::BatchNormOp;
    void tile() override;
    void run() override;

  protected:
   // This is for post-FC batch norm.
   void runNA(TiledTensor& inputs, TiledTensor& weights, TiledTensor& outputs);
   // This is for post-Conv bath norm.
   void runNHWC(TiledTensor& inputs,
                TiledTensor& weights,
                TiledTensor& outputs);

   std::array<TiledTensor, 3> tiledTensors;
};

}  // namespace smaug

#endif
