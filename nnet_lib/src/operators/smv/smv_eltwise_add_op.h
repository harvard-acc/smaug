#ifndef _OPERATORS_SMV_SMV_ELTWISE_ADD_OP_H_
#define _OPERATORS_SMV_SMV_ELTWISE_ADD_OP_H_

#include "core/backend.h"
#include "operators/common.h"
#include "operators/eltwise_add_op.h"

namespace smaug {

class SmvEltwiseAddOp : public EltwiseAddOp<SmvBackend> {
  public:
    using EltwiseAddOp<SmvBackend>::EltwiseAddOp;
    virtual void run();

  protected:
   std::array<TiledTensor, 3> doTiling();
   void runX(TiledTensor& inputs0, TiledTensor& inputs1, TiledTensor& outputs);
};


}  // namespace smaug

#endif
