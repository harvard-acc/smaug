#ifndef _OPERATORS_SMV_SMV_POOLING_OP_H_
#define _OPERATORS_SMV_SMV_POOLING_OP_H_

#include "core/backend.h"
#include "operators/common.h"
#include "operators/pooling_op.h"

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
    virtual DataLayoutSet getInputDataLayouts() const {
        return DataLayoutSet(DataLayout::NHWC);
    }
    virtual DataLayoutSet getOutputDataLayouts() const {
        return DataLayoutSet(DataLayout::NHWC);
    }
    virtual void run();
    friend class smv::pool::TilingOptimizer;

   protected:
    void runNHC(TiledTensor& inputs, TiledTensor& outputs);
};

class SmvMaxPoolingOp : public SmvPoolingOp {
   public:
    SmvMaxPoolingOp(const std::string& name, Workspace* workspace)
            : SmvPoolingOp(name, OpType::MaxPooling, workspace){};
    virtual void run();
    virtual void printSummary(std::ostream& out) const {
        const TensorShape& outputShape =
                this->outputs.at(Outputs)->getShape();
        out << this->name << " (MaxPooling)\t\t" << outputShape << "\n";
    }
};

class SmvAvgPoolingOp : public SmvPoolingOp {
   public:
    SmvAvgPoolingOp(const std::string& name, Workspace* workspace)
            : SmvPoolingOp(name, OpType::AveragePooling, workspace){};
    virtual void run();
    virtual void printSummary(std::ostream& out) const {
        const TensorShape& outputShape =
                this->outputs.at(Outputs)->getShape();
        out << this->name << " (AvgPooling)\t\t" << outputShape << "\n";
    }
};

}  // namespace smaug

#endif

