#ifndef _OPERATORS_SMV_SMV_INNER_PRODUCT_TILING_H_
#define _OPERATORS_SMV_SMV_INNER_PRODUCT_TILING_H_

#include "core/backend.h"
#include "core/tensor.h"
#include "operators/smv/smv_tiling_common.h"
#include "operators/smv/smv_tiling_base.h"

namespace smaug {

class SmvInnerProductOp;

namespace smv {
namespace fc {

class TilingOptimizer : public TilingOptimizerBase {
   public:
    static std::array<TiledTensor, 3> doTiling(SmvInnerProductOp* op);
    static TilingConfig computeBasicTileShapes(SmvInnerProductOp* op);

   protected:
    static std::array<TilingDims, 3> determineBestTilingDims(Tensor* inputs,
                                                             Tensor* weights,
                                                             Tensor* outputs,
                                                             int maxTileSize);
};

}  // namespace fc
}  // namespace smv
}  // namespace smaug

#endif
