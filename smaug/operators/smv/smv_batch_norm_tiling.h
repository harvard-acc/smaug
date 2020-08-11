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

/**
 * Tiling optimizer for SMV batch norm kernel.
 */
class TilingOptimizer : TilingOptimizerBase {
   public:
     /**
      * Runs the tiling optimizer on the given batch norm op.
      *
      * @returns An array of tiled inputs, weights, and outputs.
      */
    static std::array<TiledTensor, 3> doTiling(SmvBatchNormOp* op);

    /**
     * Determine the best basic tiling shape for this batch norm layer.
     *
     * The algorithm first determines the dimensions along which the inputs,
     * weights, and outputs will be tiled. Then based on those dimensions, we
     * enumerate all possible basic tile shapes for inputs, weights, and
     * outputs. A **basic** shape is the shape that all but potentially the
     * last tile along a set of dimensions will use. This triplet of tile
     * shapes defines a TilingConfig. The TilingConfig that maximizes the total
     * combined size of input, weights, and output tiles is chosen as the best.
     *
     * This algorithm assumes that the maximum tile size for weights, inputs,
     * and outputs are all the same and that they will reside in separate
     * scratchpads (no sharing).
     *
     * @param inputs Inputs tensor of the batch norm operator.
     * @param weights A tensor that concatenates the four weights tensors of
     * the batch norm operator.
     * @param outputs Outputs tensor of the batch norm operator.
     * @returns The TilingConfig that describes the best tiling shapes.
     */
    static TilingConfig computeBasicTileShapes(Tensor* inputs,
                                               Tensor* weights,
                                               Tensor* outputs);

   protected:
    /**
     * Determine the best tiling dimensions for running batch norm on SMV.
     *
     * Returns:
     *   A 2-element array of TilingDims enums (inputs, weights).
     */
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
