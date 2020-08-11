#ifndef _OPERATORS_SMV_SMV_INNER_PRODUCT_OP_H_
#define _OPERATORS_SMV_SMV_INNER_PRODUCT_OP_H_

#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/inner_product_op.h"

namespace smaug {

namespace smv {

/** Contains implementations of inner product on SMV and related functions. */
namespace fc {

extern const int kNumPEs;
extern const int kNumMaccsPerPE;

class TilingOptimizer;

}  // namespace fc
}  // namespace smv

/**
 * Inner product operator on SMV.
 *
 * SMV supports `C = A x B_tranpose`. Elements are 8-way vectorized.
 */
class SmvInnerProductOp : public InnerProductOp<SmvBackend> {
  public:
    using InnerProductOp<SmvBackend>::InnerProductOp;
    void tile() override;
    void run() override;
    friend class smv::fc::TilingOptimizer;

  protected:
   void runNWA(TiledTensor& inputs, TiledTensor& weights, TiledTensor& outputs);

   std::array<TiledTensor, 3> tiledTensors;
};

}  // namespace smaug

#endif
