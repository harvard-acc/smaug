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
                                         int minN,
                                         int minH,
                                         int minC);
};

}  // namespace smv
}  // namespace smaug

#endif
