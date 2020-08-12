#ifndef _OPERATORS_SMV_SMV_UNARY_OP_H_
#define _OPERATORS_SMV_SMV_UNARY_OP_H_

#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/unary_op.h"

namespace smaug {
namespace smv {

/** Contains common functions for working with unary operators. */
namespace unary {

/**
 * Extract activation function parameters from the Operator and stores them in
 * the C-style structs for passing to Aladdin.
 */
std::pair<activation_type, activation_param_t> getActivationParams(
        UnaryOp<SmvBackend>* op);

/** 
 * A generic tile dispatcher for unary operators.
 * 
 * "X" indicates that tiles can be scheduled in any order.
 */
void runX(UnaryOp<SmvBackend>* op, TiledTensor& inputs, TiledTensor& outputs);

/**
 * Tile the provided Tensor.
 *
 * This is only for unary operators. The only requirement is to tile the Tensor
 * in contiguous blocks of tileShape.
 */
TiledTensor generateTiles(Tensor* tensor,
                          const TensorShape& tileShape,
                          Operator* op,
                          bool copyData = true);

std::array<TiledTensor, 2> doTiling(UnaryOp<SmvBackend>* op,
                                    bool copyData = true);

void run(UnaryOp<SmvBackend>* op, std::array<TiledTensor, 2>& tiledTensors);

}  // namespace unary
}  // namespace smv
}  // namespace smaug

#endif

