#ifndef _OPERATORS_SMV_SMV_TILING_BASE_H_
#define _OPERATORS_SMV_SMV_TILING_BASE_H_

#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/operators/smv/smv_tiling_common.h"

namespace smaug {
namespace smv {

class TilingOptimizerBase {
   protected:

    /**
     * Find the best set of dimensions to tile a given tensor shape.
     *
     * The goal is to divide up a tensor into tiles that each are <=
     * maxTileSize elements. The tensor layouts can be NC, NHWC or NCHW. The
     * minimum tile size is specified via minShape. The preferences for tiling
     * dimensions are as follows:
     *
     * 1) No tiling.
     * 2) Dim-N tiling. N is the input batch.
     * 3) Dim-NC tiling. After tiling by N, tile channelwise. Do not tile in HW.
     * 4) Dim-NH tiling. After tiling by N, tile rowwise. Do not tile in WC.
     * 5) Dim-NW tiling. After tiling by N, tile columnwise. Do not tile in HC.
     * 6) Dim-NHW tiling. After tiling by N, tile rowwise and then columnwise.
     *    Do not tile in C.
     * 7) Dim-NCH tiling. After tiling by N and channel dimensions, tile
     *    rowwise. Do not tile in W.
     * 8) Dim-NCW tiling. After tiling by N and channel dimensions, tile
     *    columnwise. Do not tile in H.
     *
     * Except Option 1, a minimum size for each dimension can be deduced from
     * minShape. Note that for a 2D tensor shape, only options 1-5 are viable.
     */
    static TilingDims findBestTilingDims(const TensorShape& shape,
                                         int maxTileSize,
                                         const std::vector<int>& minShape);

    /**
     * Enumerates all tiling configs for a two dimensional Tensor.
     *
     * @param shape Tensor shape.
     * @param maxTileSize Maximum elements per tile.
     * @param minShape Minimum per-tile shape
     * @param strides Stride lengths along the N and C channels.
     * @param configs Output list of tiling configurations.
     */
    static void enum2DTensorTilingConfigs(TensorShape shape,
                                          int maxTileSize,
                                          const std::vector<int>& minShape,
                                          const std::vector<int>& strides,
                                          std::vector<TensorShape>& configs);
    /**
     * Enumerates all tiling configs for a four dimensional Tensor.
     *
     * @param shape Tensor shape.
     * @param maxTileSize Maximum elements per tile.
     * @param minShape Minimum per-tile shape
     * @param strides Stride lengths along the N and C channels.
     * @param configs Output list of tiling configurations.
     */
    static void enum4DTensorTilingConfigs(TensorShape shape,
                                          int maxTileSize,
                                          const std::vector<int>& minShape,
                                          const std::vector<int>& strides,
                                          std::vector<TensorShape>& configs);
};

}  // namespace smv
}  // namespace smaug

#endif
