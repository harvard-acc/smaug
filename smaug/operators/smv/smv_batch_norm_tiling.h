#ifndef _OPERATORS_SMV_SMV_BATCH_NORM_TILING_H_
#define _OPERATORS_SMV_SMV_BATCH_NORM_TILING_H_

#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/operators/smv/smv_tiling_common.h"
#include "smaug/operators/smv/smv_tiling_base.h"

namespace smaug {

class SmvBatchNormOp;

namespace smv {
namespace bn {

class TilingOptimizer : TilingOptimizerBase {
   public:
    static std::array<TiledTensor, 3> doTiling(SmvBatchNormOp* op);
    static TilingConfig computeBasicTileShapes(Tensor* inputs,
                                               Tensor* weights,
                                               Tensor* outputs);

   protected:
    static std::array<TilingDims, 2> determineBestTilingDims(Tensor* inputs,
                                                             Tensor* weights,
                                                             int maxTileSize);
    static void enumPostFCTilingConfigs(TensorShape inputsShape,
                                        TensorShape weightsShape,
                                        int maxTileSize,
                                        std::array<TilingDims, 2> strategies,
                                        std::list<TilingConfig>& fullConfigs);
    static void enumPostConvTilingConfigs(TensorShape inputsShape,
                                          TensorShape weightsShape,
                                          int maxTileSize,
                                          std::array<TilingDims, 2> strategies,
                                          std::list<TilingConfig>& fullConfigs);
};

}  // namespace bn
}  // namespace smv
}  // namespace smaug

#endif
