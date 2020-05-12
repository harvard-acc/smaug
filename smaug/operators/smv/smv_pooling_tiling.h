#ifndef _OPERATORS_SMV_SMV_POOLING_TILING_H_
#define _OPERATORS_SMV_SMV_POOLING_TILING_H_

#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/operators/smv/smv_tiling_common.h"
#include "smaug/operators/smv/smv_tiling_base.h"

namespace smaug {

class SmvPoolingOp;

namespace smv {
namespace pool {

class TilingOptimizer : public TilingOptimizerBase {
   public:
    static std::array<TiledTensor, 2> doTiling(SmvPoolingOp* op);
    static TilingConfig computeBasicTileShapes(SmvPoolingOp* op);

   protected:
    static std::array<TilingDims, 2> determineBestTilingDims(
            Tensor* inputs,
            Tensor* outputs,
            int maxTileSize,
            std::pair<int, int> poolSize);
};

}  // namespace pool
}  // namespace smv
}  // namespace smaug

#endif
