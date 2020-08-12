#ifndef _OPERATORS_SMV_SMV_POOLING_OP_H_
#define _OPERATORS_SMV_SMV_POOLING_OP_H_

#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/pooling_op.h"

namespace smaug {

namespace smv {

/** Contains pooling operators and tiling optimizers for SMV. */
namespace pool {

extern const int kVectorSize;

class TilingOptimizer;

}  // namespace pool
}  // namespace smv

/** Base class for SMV pooling oeprators */
class SmvPoolingOp : public PoolingOp<SmvBackend> {
   public:
    using PoolingOp<SmvBackend>::PoolingOp;
    void tile() override;
    void run() override;
    friend class smv::pool::TilingOptimizer;

   protected:
    void runNHWC(TiledTensor& inputs, TiledTensor& outputs);

    std::array<TiledTensor, 2> tiledTensors;
};

/** Max-pooling operator on SMV. */
class SmvMaxPoolingOp : public SmvPoolingOp {
   public:
    SmvMaxPoolingOp(const std::string& name, Workspace* workspace)
            : SmvPoolingOp(name, OpType::MaxPooling, workspace){};
    void tile() override;
    void run() override;
};

/** Average pooling operator on SMV. */
class SmvAvgPoolingOp : public SmvPoolingOp {
   public:
    SmvAvgPoolingOp(const std::string& name, Workspace* workspace)
            : SmvPoolingOp(name, OpType::AveragePooling, workspace){};
    void tile() override;
    void run() override;
};

}  // namespace smaug

#endif

