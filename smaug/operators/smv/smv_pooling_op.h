#ifndef _OPERATORS_SMV_SMV_POOLING_OP_H_
#define _OPERATORS_SMV_SMV_POOLING_OP_H_

#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/pooling_op.h"

namespace smaug {

namespace smv {
namespace pool {

extern const int kVectorSize;

class TilingOptimizer;

}  // namespace pool
}  // namespace smv

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

class SmvMaxPoolingOp : public SmvPoolingOp {
   public:
    SmvMaxPoolingOp(const std::string& name, Workspace* workspace)
            : SmvPoolingOp(name, OpType::MaxPooling, workspace){};
    void tile() override;
    void run() override;
};

class SmvAvgPoolingOp : public SmvPoolingOp {
   public:
    SmvAvgPoolingOp(const std::string& name, Workspace* workspace)
            : SmvPoolingOp(name, OpType::AveragePooling, workspace){};
    void tile() override;
    void run() override;
};

}  // namespace smaug

#endif

