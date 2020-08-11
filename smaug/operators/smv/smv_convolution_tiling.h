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

/**
 * Tiling optimizer for SMV convolution kernel.
 */
class TilingOptimizer : public TilingOptimizerBase {
   public:
    static std::array<TiledTensor, 3> doTiling(SmvConvolutionOp* op);

    /**
     * Determine the best basic tiling shape for this convolution layer.
     *
     * The algorithm first determines the dimensions along which the inputs,
     * weights, and outputs will be tiled. Then based on those dimensions, we
     * enumerate all possible basic tile shapes for inputs, weights, and
     * outputs. A **basic** shape is the shape that all but potentially the
     * last tile along a set of dimensions will use. This triplet of tile
     * shapes defines a TilingConfig. The TilingConfig that maximizes the total
     * combined size of input, weights, and output tiles is chosen as the best.
     *
     * To limit the number of possibilities, we only enumerate each dimension
     * in certain increments. For example, input channels are only enumerated
     * in multiples of kNumMaccsPerPE, and output channels are only enumerated
     * in multiples in kNumPEs.
     *
     * This algorithm assumes that the maximum tile size for weights, inputs,
     * and outputs are all the same and that they will reside in separate
     * scratchpads (no sharing).
     *
     * @param op The SMV convolution operator. All tensors must have been
     * created with createAllTensors() prior to calling this function.
     * @returns The TilingConfig that describes the best tiling shapes.
     */
    static TilingConfig computeBasicTileShapes(SmvConvolutionOp* op);

    /**
     * A specialized output tiling function when the output is tiled rowwise.
     *
     * This accounts for additional corner cases in filter field size and
     * zero-padding that arise with specific inputs/weights tile sizes.
     */
    static TiledTensor generateRowwiseOutputTiledTensor(
            SmvConvolutionOp* op,
            const TiledTensor& inputTiledTensor,
            const TiledTensor& weightsTiledTensor,
            const TensorShape& maxOutputTileSize,
            Tensor* outputTensor,
            bool copyData = false);

   protected:
    /**
     * Determine the best tiling dimensions for running convolution on SMV.
     *
     * This function imposes some additional constraints on the tiling
     * dimensions, in that certain combinations of input/weight/output tiling
     * dimensions are not allowed in the interest of tiling code complexity.
     *
     * @returns A 3-element array of TilingDims enums (inputs, weights,
     * outputs).
     */
    static std::array<TilingDims, 3> determineBestTilingDims(Tensor* inputs,
                                                             Tensor* weights,
                                                             Tensor* outputs,
                                                             int maxTileSize);
};

}  // namespace conv
}  // namespace smv
}  // namespace smaug

#endif
