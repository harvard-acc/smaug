#ifndef _OPERATORS_SMV_SMV_CONVOLUTION_TILING_H_
#define _OPERATORS_SMV_SMV_CONVOLUTION_TILING_H_

#include "core/backend.h"
#include "core/tensor.h"
#include "operators/smv/tiling_common.h"

namespace smaug {

class SmvConvolutionOp;

namespace smv {
namespace conv {

class TilingOptimizer {
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
    static TilingDims findBestTilingDims(const TensorShape& shape,
                                         int maxTileSize,
                                         int minN,
                                         int minH,
                                         int minC);
    static std::array<TilingDims, 3> determineBestTilingDims(Tensor* inputs,
                                                             Tensor* weights,
                                                             Tensor* outputs,
                                                             int maxTileSize);
};

}  // namespace conv
}  // namespace smv
}  // namespace smaug

#endif
