#ifndef _OPERATORS_SMV_SMV_GREATER_OP_H_
#define _OPERATORS_SMV_SMV_GREATER_OP_H_

#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/greater_op.h"

namespace smaug {

/** Elementwise greater-than operator on SMV. */
class SmvGreaterOp : public GreaterOp<SmvBackend> {
  public:
    using GreaterOp<SmvBackend>::GreaterOp;
    void tile() override;
    void run() override;

  protected:
   void runX(TiledTensor& inputs0, TiledTensor& inputs1, TiledTensor& outputs);

   std::array<TiledTensor, 3> tiledTensors;
};

/** Elementwise greater-than-or-equal-to operator on SMV. */
class SmvGreaterEqualOp : public GreaterEqualOp<SmvBackend> {
  public:
    using GreaterEqualOp<SmvBackend>::GreaterEqualOp;
    void tile() override;
    void run() override;

  protected:
   void runX(TiledTensor& inputs0, TiledTensor& inputs1, TiledTensor& outputs);

   std::array<TiledTensor, 3> tiledTensors;
};


}  // namespace smaug

#endif
