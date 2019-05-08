#ifndef _OPERATORS_SMV_SMV_TILING_BASE_H_
#define _OPERATORS_SMV_SMV_TILING_BASE_H_

#include "core/backend.h"
#include "core/tensor.h"
#include "operators/smv/smv_tiling_common.h"

namespace smaug {
namespace smv {

class TilingOptimizerBase {
   protected:
    static TilingDims findBestTilingDims(const TensorShape& shape,
                                         int maxTileSize,
                                         const std::vector<int>& minShape);
    static void enum2DTensorTilingConfigs(TensorShape shape,
                                          int maxTileSize,
                                          const std::vector<int>& minShape,
                                          const std::vector<int>& strides,
                                          std::vector<TensorShape>& configs);
    static void enum4DTensorTilingConfigs(TensorShape shape,
                                          int maxTileSize,
                                          const std::vector<int>& minShape,
                                          const std::vector<int>& strides,
                                          std::vector<TensorShape>& configs);
};

}  // namespace smv
}  // namespace smaug

#endif
