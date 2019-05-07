#include "core/backend.h"
#include "core/tensor_utils.h"
#include "operators/common.h"
#include "operators/smv/smv_tiling_base.h"
#include "utility/debug_stream.h"

namespace smaug {
namespace smv {

// Find the best set of dimensions to tile a given tensor shape.
//
// The goal is to divide up a tensor into tiles that each are <= maxTileSize
// elements. Assuming that the input tensor data layout is NHWC, the
// preferences for tiling dimensions are as follows:
//   1) No tiling.
//   2) Dim-N tiling. N is the input batch.
//   3) Dim-NC tiling. After tiling by N, tile channelwise. Do not tile in HW.
//   4) Dim-NH tiling. After tiling by N, tile rowwise. Do not tile in WC.
//   5) Dim-NCH tiling. After tiling by N and channel dimensions, tile rowwise.
//      Do not tile in W.
//
// For options 2-5, a minimum size for each dimension can be specified via
// minN, minH, and minC. Note that for a 2D tensor shape, only options 1-3 are
// viable.
//
// TODO: W-wise (or column-wise) tiling is not supported yet.
TilingDims TilingOptimizerBase::findBestTilingDims(const TensorShape& shape,
                                                   int maxTileSize,
                                                   int minN,
                                                   int minH,
                                                   int minC) {
    if (shape.storageSize() <= maxTileSize)
        return TilingDims::None;
    int ndims = shape.ndims();
    minN = std::min(shape[0], minN);
    // C is the last dimension, so we need to apply padding.
    minC = std::min(shape.getStorageDim(ndims - 1), minC);
    int sizePerN = shape.storageSize() / shape[0];
    if (sizePerN * minN <= maxTileSize)
        return TilingDims::DimN;
    if (sizePerN * (minC * 1.0 / shape[ndims - 1]) <= maxTileSize)
        return TilingDims::DimNC;
    if (shape.ndims() == 2) {
        std::cerr << "[ERROR]: Unable to find a supported set of tiling "
                     "dimensions for 2D tensor with shape "
                  << shape << "!\n";
        exit(1);
    }
    minH = std::min(shape[1], minH);
    if (sizePerN * (minH * 1.0 / shape[1]) <= maxTileSize)
        return TilingDims::DimNH;
    if (sizePerN * (minC * 1.0 / shape[3]) * (minH * 1.0 / shape[1]) <=
        maxTileSize)
        return TilingDims::DimNCH;
    std::cerr << "[ERROR]: Unable to find a supported set of tiling dimensions "
                 "for 4D tensor with shape "
              << shape << "!\n";
    exit(1);
}

}  // namespace smv
}  // namespace smaug
