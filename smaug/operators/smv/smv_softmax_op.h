#ifndef _OPERATORS_SMV_SMV_SOFTMAX_OP_H_
#define _OPERATORS_SMV_SMV_SOFTMAX_OP_H_

#include "core/backend.h"
#include "operators/common.h"
#include "operators/softmax_op.h"

namespace smaug {

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
