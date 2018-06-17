#ifndef _OPERATORS_SMV_SMV_CONVOLUTION_TILING_H_
#define _OPERATORS_SMV_SMV_CONVOLUTION_TILING_H_

#include "core/backend.h"
#include "core/tensor.h"

namespace smaug {

class SmvConvolutionOp;
typedef Tensor<SmvBackend> SmvTensor;

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
    static std::vector<SmvTensor*> generateBlockedTensor(
            SmvTensor* tensor, const TensorShape& tileShape);

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
