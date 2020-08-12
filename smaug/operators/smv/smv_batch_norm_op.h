#ifndef _OPERATORS_SMV_SMV_BATCH_NORM_OP_H_
#define _OPERATORS_SMV_SMV_BATCH_NORM_OP_H_

#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/batch_norm_op.h"

namespace smaug {

namespace smv {

/** Contains batch-norm implementations and tiling optimizers for SMV. */
namespace bn {

extern const int kVectorSize;

class TilingOptimizer;

}  // namespace bn
}  // namespace smv

/**
 * SMV backend implementation of batch normalization.
 *
 * Elements are formatted and consumed in vectors of 8.
 */
class SmvBatchNormOp : public BatchNormOp<SmvBackend> {
  public:
    using BatchNormOp<SmvBackend>::BatchNormOp;
    void tile() override;
    void run() override;

  protected:
   /** Post-FC tile dispatcher. */
   void runNA(TiledTensor& inputs, TiledTensor& weights, TiledTensor& outputs);

   /** Post-convolution tile dispatcher. */
   void runNHWC(TiledTensor& inputs,
                TiledTensor& weights,
                TiledTensor& outputs);

   std::array<TiledTensor, 3> tiledTensors;
};

}  // namespace smaug

#endif
