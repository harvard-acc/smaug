#ifndef _OPERATORS_SMV_SMV_CONVOLUTION_TILING_H_
#define _OPERATORS_SMV_SMV_CONVOLUTION_TILING_H_

#include "core/backend.h"
#include "core/tensor.h"

namespace smaug {

class SmvConvolutionOp;

namespace smv {
namespace conv {

enum TilingDims {
    None,
    DimN,
    DimNC,
    DimNH,
    Invalid
};

struct TilingConfig {
  public:
    TilingConfig() {}

    int getTotalSize() const {
        return inputs.storageSize() + weights.storageSize() +
               outputs.storageSize();
    }

    TensorShape inputs;
    TensorShape weights;
    TensorShape outputs;
};

class TilingOptimizer {
   public:
    static TilingConfig computeBasicTileShapes(SmvConvolutionOp* op);
    static SmvTiledTensor generateTiledTensor(SmvTensor* tensor,
                                              const TensorShape& tileShape,
                                              std::vector<int> halos);
    static SmvTiledTensor generateDimNHOutputTiledTensor(
            SmvConvolutionOp* op,
            const SmvTiledTensor& inputTiledTensor,
            const SmvTiledTensor& weightsTiledTensor,
            const TensorShape& maxOutputTileSize,
            SmvTensor* outputTensor,
            bool copyData = false);

   protected:
    static TilingDims findBestTilingDims(const TensorShape& shape,
                                         int maxTileSize,
                                         int minN,
                                         int minH,
                                         int minC);
    static std::array<TilingDims, 3> determineBestTilingDims(SmvTensor* inputs,
                                                             SmvTensor* weights,
                                                             SmvTensor* outputs,
                                                             int maxTileSize);
};

}  // namespace conv
}  // namespace smv
}  // namespace smaug

#endif
