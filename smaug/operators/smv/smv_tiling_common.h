#ifndef _OPERATORS_SMV_TILING_COMMON_H_
#define _OPERATORS_SMV_TILING_COMMON_H_

#include "smaug/core/tensor.h"

namespace smaug {
namespace smv {

/**
 * The set of supported tiling strategies. Each strategy indicates along which
 * set of dimensions a Tensor should be tiled along.
 */
enum TilingDims {
    None,
    DimN,
    DimNC,
    DimNH,
    DimNW,
    DimNHW,
    DimNCH,
    DimNCW,
    Invalid
};

/**
 * A TilingConfig describes tiling strategies and optimal tile sizes for inputs,
 * weights, and outputs Tensors. 
 */
struct TilingConfig {
  public:
    TilingConfig(TensorShape _inputs = TensorShape(),
                 TensorShape _weights = TensorShape(),
                 TensorShape _outputs = TensorShape())
            : inputs(_inputs), weights(_weights), outputs(_outputs) {}

    int getTotalSize() const {
        return inputs.storageSize() + weights.storageSize() +
               outputs.storageSize();
    }

    TensorShape inputs;
    TensorShape weights;
    TensorShape outputs;
    TilingDims inputTilingDims;
    TilingDims weightTilingDims;
    TilingDims outputTilingDims;
};

std::ostream& operator<<(std::ostream& os, const TilingDims& dims);
std::ostream& operator<<(std::ostream& os, const TilingConfig& config);

bool needsNwiseTiling(TilingDims dim);

bool needsCwiseTiling(TilingDims dim);

bool needsHwiseTiling(TilingDims dim);

bool needsWwiseTiling(TilingDims dim);

}  // namespace smv
}  // namespace smaug

#endif
