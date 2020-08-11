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

/**
 * Tiling optimizer for pooling operators on SMV. 
 */
class TilingOptimizer : public TilingOptimizerBase {
   public:
    static std::array<TiledTensor, 2> doTiling(SmvPoolingOp* op);

    /**
     * Determine the best basic tiling shape for this pooling layer.
     *
     * The algorithm first determines the dimensions along which the inputs and
     * outputs will be tiled. Then based on those dimensions, we enumerate all
     * possible basic tile shapes for inputs and outputs. A **basic** shape is
     * the shape that all but potentially the last tile along a set of
     * dimensions will use. This duo of tile shapes defines a TilingConfig. The
     * TilingConfig that maximizes the total combined size of input and output
     * tiles is chosen as the best.
     *
     * To limit the number of possibilities, we only enumerate each dimension
     * in certain increments. For example, input channels are only enumerated
     * in multiples of kVectorSize.
     *
     * This algorithm assumes that the maximum tile size for inputs and outputs
     * are all the same and that they will reside in separate scratchpads (no
     * sharing).
     *
     * @param op The SMV pooling operator. All tensors must have been created
     * with createAllTensors() prior to calling this function.
     * @returns The TilingConfig that describes the best tiling shapes.
     */
    static TilingConfig computeBasicTileShapes(SmvPoolingOp* op);

   protected:

    /**
     * Determine the best tiling dimensions for running pooling on SMV.
     *
     * This function imposes some additional constraints on the tiling
     * dimensions, in that certain combinations of input/output tiling
     * dimensions are not allowed in the interest of tiling code complexity.
     *
     * @returns A 2-element array of TilingDims enums (inputs, outputs).
     */
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
