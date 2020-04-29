#ifndef _OPERATORS_SMV_SMV_UNARY_OP_H_
#define _OPERATORS_SMV_SMV_UNARY_OP_H_

#include "core/backend.h"
#include "operators/common.h"
#include "operators/unary_op.h"

namespace smaug {
namespace smv {
namespace unary {

std::pair<activation_type, activation_param_t> getActivationParams(
        UnaryOp<SmvBackend>* op);

void runX(UnaryOp<SmvBackend>* op, TiledTensor& inputs, TiledTensor& outputs);

TiledTensor generateTiles(Tensor* tensor,
                          const TensorShape& tileShape,
                          Operator* op,
                          bool copyData = true);

std::array<TiledTensor, 2> doTiling(UnaryOp<SmvBackend>* op,
                                    bool copyData = true);

void run(UnaryOp<SmvBackend>* op);

}  // namespace unary
}  // namespace smv
}  // namespace smaug

#endif

