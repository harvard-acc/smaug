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
// elements. The tensor layouts can be NC, NHWC or NCHW. The minimum tile size
// is specified via minShape. The preferences for tiling dimensions are as
// follows:
//   1) No tiling.
//   2) Dim-N tiling. N is the input batch.
//   3) Dim-NC tiling. After tiling by N, tile channelwise. Do not tile in HW.
//   4) Dim-NH tiling. After tiling by N, tile rowwise. Do not tile in WC.
//   5) Dim-NCH tiling. After tiling by N and channel dimensions, tile rowwise.
//      Do not tile in W.
//
// For options 2-5, a minimum size for each dimension can be deduced from
// minShape. Note that for a 2D tensor shape, only options 1-3 are viable.
//
// TODO: W-wise (or column-wise) tiling is not supported yet.
TilingDims TilingOptimizerBase::findBestTilingDims(
        const TensorShape& shape,
        int maxTileSize,
        const std::vector<int>& minShape) {
    DataLayout layout = shape.getLayout();
    assert(layout == NHWC || layout == NCHW || layout == NC);
    if (shape.storageSize() <= maxTileSize)
        return TilingDims::None;
    int minN = std::min(shape[0], minShape[0]);
    bool isNHWC = layout == NHWC;
    int cIdx = isNHWC ? 3 : 1;
    int minC = std::min(shape[cIdx], minShape[cIdx]);
    int sizePerN = shape.storageSize() / shape[0];
    if (sizePerN * minN <= maxTileSize)
        return TilingDims::DimN;
    if (sizePerN * (minC * 1.0 / shape[cIdx]) <= maxTileSize)
        return TilingDims::DimNC;
    if (shape.ndims() == 2) {
        std::cerr << "[ERROR]: Unable to find a supported set of tiling "
                     "dimensions for 2D tensor with shape "
                  << shape << "!\n";
        exit(1);
    }
    int hIdx = isNHWC ? 1 : 2;
    int minH = std::min(shape[hIdx], minShape[hIdx]);
    if (sizePerN * (minH * 1.0 / shape[hIdx]) <= maxTileSize)
        return TilingDims::DimNH;
    if (sizePerN * (minC * 1.0 / shape[cIdx]) * (minH * 1.0 / shape[hIdx]) <=
        maxTileSize)
        return TilingDims::DimNCH;
    std::cerr << "[ERROR]: Unable to find a supported set of tiling dimensions "
                 "for 4D tensor with shape "
              << shape << "!\n";
    exit(1);
}

void TilingOptimizerBase::enum2DTensorTilingConfigs(
        TensorShape shape,
        int maxTileSize,
        const std::vector<int>& minShape,
        const std::vector<int>& strides,
        std::vector<TensorShape>& configs) {
    int minN = std::min(minShape[0], shape[0]);
    int minC = std::min(minShape[1], shape[1]);
    int strideN = strides[0];
    int strideC = strides[1];
    for (int n = minN; n <= shape[0]; n += strideN) {
        for (int c = minC; c <= shape[1]; c += strideC) {
            TensorShape config(
                    { n, c }, shape.getLayout(), shape.getAlignment());
            if (config.storageSize() <= maxTileSize)
                configs.push_back(config);
            else
                break;
        }
    }
}

void TilingOptimizerBase::enum4DTensorTilingConfigs(
        TensorShape shape,
        int maxTileSize,
        const std::vector<int>& minShape,
        const std::vector<int>& strides,
        std::vector<TensorShape>& configs) {
    bool isNHWC = shape.getLayout() == NHWC;
    int idxH = isNHWC ? 1 : 2;
    int idxW = isNHWC ? 2 : 3;
    int idxC = isNHWC ? 3 : 1;
    int minN = std::min(minShape[0], shape[0]);
    int minH = std::min(minShape[idxH], shape[idxH]);
    int minW = std::min(minShape[idxW], shape[idxW]);
    int minC = std::min(minShape[idxC], shape[idxC]);
    int strideN = strides[0];
    int strideH = strides[idxH];
    int strideW = strides[idxW];
    int strideC = strides[idxC];
    for (int n = minN; n <= shape[0]; n += strideN) {
        for (int c = minC; c <= shape[idxC]; c += strideC) {
            for (int h = minH; h <= shape[idxH]; h += strideH) {
                for (int w = minW; w <= shape[idxW]; w += strideW) {
                    TensorShape config;
                    if (isNHWC) {
                        config = TensorShape({ n, h, w, c },
                                             shape.getLayout(),
                                             shape.getAlignment());
                    } else {
                        config = TensorShape({ n, c, h, w },
                                             shape.getLayout(),
                                             shape.getAlignment());
                    }
                    if (config.storageSize() <= maxTileSize)
                        configs.push_back(config);
                    else
                        break;
                }
            }
        }
    }
}

}  // namespace smv
}  // namespace smaug
