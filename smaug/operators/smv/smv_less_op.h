#ifndef _OPERATORS_SMV_SMV_LESS_OP_H_
#define _OPERATORS_SMV_SMV_LESS_OP_H_

#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/less_op.h"

namespace smaug {

/** Elementwise less-than operator on SMV. */
class SmvLessOp : public LessOp<SmvBackend> {
  public:
    using LessOp<SmvBackend>::LessOp;
    void tile() override;
    void run() override;

  protected:
   void runX(TiledTensor& inputs0, TiledTensor& inputs1, TiledTensor& outputs);

   std::array<TiledTensor, 3> tiledTensors;
};

/** Elementwise less-than-or-equal-to operator on SMV. */
class SmvLessEqualOp : public LessEqualOp<SmvBackend> {
  public:
    using LessEqualOp<SmvBackend>::LessEqualOp;
    void tile() override;
    void run() override;

  protected:
   void runX(TiledTensor& inputs0, TiledTensor& inputs1, TiledTensor& outputs);

   std::array<TiledTensor, 3> tiledTensors;
};


}  // namespace smaug

#endif
