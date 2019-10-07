#ifndef _OPERATORS_SMV_SMV_INNER_PRODUCT_OP_H_
#define _OPERATORS_SMV_SMV_INNER_PRODUCT_OP_H_

#include "core/backend.h"
#include "operators/common.h"
#include "operators/inner_product_op.h"

namespace smaug {

namespace smv {
namespace fc {

extern const int kNumPEs;
extern const int kNumMaccsPerPE;

class TilingOptimizer;

}  // namespace fc
}  // namespace smv

class SmvInnerProductOp : public InnerProductOp<SmvBackend> {
  public:
    using InnerProductOp<SmvBackend>::InnerProductOp;
    void run() override;
    friend class smv::fc::TilingOptimizer;

  protected:
   void runNWA(TiledTensor& inputs, TiledTensor& weights, TiledTensor& outputs);
};

}  // namespace smaug

#endif
