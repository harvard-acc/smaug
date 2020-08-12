#ifndef _OPERATORS_SMV_SMV_ELTWISE_ADD_OP_H_
#define _OPERATORS_SMV_SMV_ELTWISE_ADD_OP_H_

#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/eltwise_add_op.h"

namespace smaug {

/** Elementwise addition on SMV. */
class SmvEltwiseAddOp : public EltwiseAddOp<SmvBackend> {
  public:
    using EltwiseAddOp<SmvBackend>::EltwiseAddOp;
    void tile() override;
    void run() override;

  protected:
   void runX(TiledTensor& inputs0, TiledTensor& inputs1, TiledTensor& outputs);

   std::array<TiledTensor, 3> tiledTensors;
};


}  // namespace smaug

#endif
