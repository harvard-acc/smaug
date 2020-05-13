#ifndef _OPERATORS_SMV_SMV_ELTWISE_MUL_OP_H_
#define _OPERATORS_SMV_SMV_ELTWISE_MUL_OP_H_

#include "core/backend.h"
#include "operators/common.h"
#include "operators/eltwise_mul_op.h"

namespace smaug {

class SmvEltwiseMulOp : public EltwiseMulOp<SmvBackend> {
  public:
    using EltwiseMulOp<SmvBackend>::EltwiseMulOp;
    void tile() override;
    void run() override;

  protected:
   void runX(TiledTensor& inputs0, TiledTensor& inputs1, TiledTensor& outputs);

   std::array<TiledTensor, 3> tiledTensors;
};


}  // namespace smaug

#endif
