#ifndef _OPERATORS_SMV_SMV_SOFTMAX_OP_H_
#define _OPERATORS_SMV_SMV_SOFTMAX_OP_H_

#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/softmax_op.h"

namespace smaug {

/** Softmax operator on SMV. */
class SmvSoftmaxOp : public SoftmaxOp<SmvBackend> {
   public:
    using SoftmaxOp<SmvBackend>::SoftmaxOp;
    void tile() override;
    void run() override;

   protected:
    std::array<TiledTensor, 2> tiledTensors;
};

}  // namespace smaug

#endif
