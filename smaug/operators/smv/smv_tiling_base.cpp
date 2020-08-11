#include "smaug/core/backend.h"
#include "smaug/core/tensor_utils.h"
#include "smaug/operators/common.h"
#include "smaug/operators/smv/smv_tiling_base.h"
#include "smaug/utility/debug_stream.h"

namespace smaug {
namespace smv {

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
    int wIdx = isNHWC ? 2 : 3;
    int minH = std::min(shape[hIdx], minShape[hIdx]);
    int minW = std::min(shape[wIdx], minShape[wIdx]);
    if (sizePerN * (minH * 1.0 / shape[hIdx]) <= maxTileSize)
        return TilingDims::DimNH;
    if (sizePerN * (minW * 1.0 / shape[wIdx]) <= maxTileSize)
        return TilingDims::DimNW;
    if (sizePerN * (minH * 1.0 / shape[hIdx]) * (minW * 1.0 / shape[wIdx]) <=
        maxTileSize)
        return TilingDims::DimNHW;
    if (sizePerN * (minC * 1.0 / shape[cIdx]) * (minH * 1.0 / shape[hIdx]) <=
        maxTileSize)
        return TilingDims::DimNCH;
    if (sizePerN * (minC * 1.0 / shape[cIdx]) * (minW * 1.0 / shape[wIdx]) <=
        maxTileSize)
        return TilingDims::DimNCW;
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
