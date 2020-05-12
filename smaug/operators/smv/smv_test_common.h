#include "smaug/core/tensor.h"

namespace smaug {

// For the operator tests, tensors should be initialized with random data so
// that more corner cases can be tested. For tiling tests, fixed data is used
// for easy verification.

void fillTensorWithRandomData(Tensor* tensor);

void fillTensorWithFixedData(Tensor* tensor);

void verifyTensorWithFixedData(Tensor* tensor, int valueOffset);

}  // namespace smaug
