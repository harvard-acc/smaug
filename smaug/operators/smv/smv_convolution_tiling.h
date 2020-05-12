#ifndef _OPERATORS_SMV_SMV_CONVOLUTION_TILING_H_
#define _OPERATORS_SMV_SMV_CONVOLUTION_TILING_H_

#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/operators/smv/smv_tiling_common.h"
#include "smaug/operators/smv/smv_tiling_base.h"

namespace smaug {

class SmvConvolutionOp;

namespace smv {
namespace conv {

class TilingOptimizer : public TilingOptimizerBase {
   public:
    static std::array<TiledTensor, 3> doTiling(SmvConvolutionOp* op);
    static TilingConfig computeBasicTileShapes(SmvConvolutionOp* op);
    static TiledTensor generateRowwiseOutputTiledTensor(
            SmvConvolutionOp* op,
            const TiledTensor& inputTiledTensor,
            const TiledTensor& weightsTiledTensor,
            const TensorShape& maxOutputTileSize,
            Tensor* outputTensor,
            bool copyData = false);

   protected:
    static std::array<TilingDims, 3> determineBestTilingDims(Tensor* inputs,
                                                             Tensor* weights,
                                                             Tensor* outputs,
                                                             int maxTileSize);
};

}  // namespace conv
}  // namespace smv
}  // namespace smaug

#endif
