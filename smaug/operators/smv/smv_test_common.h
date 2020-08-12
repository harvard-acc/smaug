#include "smaug/core/tensor.h"

namespace smaug {

// For the operator tests, tensors should be initialized with random data so
// that more corner cases can be tested. For tiling tests, fixed data is used
// for easy verification.

/** This fills the Tensor with normally distributed random values. */
void fillTensorWithRandomData(Tensor* tensor);

/** 
 * This fills the Tensor with a fixed data pattern.
 *
 * The Tensor should be in NWCH data layout. Each channel dimension is
 * initialized with a different value, but each batch/row/col will share this
 * same pattern
 */
void fillTensorWithFixedData(Tensor* tensor);

/**
 * Verify that the provided Tensor's data matches the fixed pattern produced by
 * fillTensorWithFixedData, with the provided offset to each value.
 */
void verifyTensorWithFixedData(Tensor* tensor, int valueOffset);

}  // namespace smaug
