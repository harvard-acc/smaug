#include <random>

#include "catch.hpp"
#include "smaug/core/smaug_test.h"
#include "smaug/operators/smv/smv_test_common.h"

namespace smaug {

constexpr float kMean = 0.0;
constexpr float kVar = 0.1;
std::default_random_engine generator;
std::normal_distribution<float> normalDist(kMean, kVar);
constexpr float kFraction = 0.1;

// This fills the tensor with a normal distribution of random values.
void fillTensorWithRandomData(Tensor* tensor) {
    float16* dataPtr = tensor->data<float16>();
    for (int i = 0; i < tensor->getShape().storageSize(); i++)
        dataPtr[i] = fp16(normalDist(generator));
}

void fillTensorWithFixedData(Tensor* tensor) {
    const TensorShape& shape = tensor->getShape();
    // Each dimension C is initialized to a different constant value.
    float16* dataPtr = tensor->data<float16>();
    int resetCounter = shape.getStorageDim(shape.ndims() - 1);
    int value = 0;
    for (int i = 0; i < shape.storageSize(); i++) {
        dataPtr[i] = fp16((value++) * kFraction);
        if ((i + 1) % resetCounter == 0)
            value = 0;
    }
}

void verifyTensorWithFixedData(Tensor* tensor, int valueOffset) {
    const TensorShape& shape = tensor->getShape();
    float16* dataPtr = tensor->data<float16>();
    int expectedValue = valueOffset;
    int resetCounter = tensor->getShape().getStorageDim(shape.ndims() - 1);
    int totalSize = tensor->getShape().storageSize();
    for (int i = 0; i < totalSize; i++) {
        REQUIRE(Approx(fp32(dataPtr[i])).epsilon(kEpsilon) ==
                expectedValue * kFraction);
        ++expectedValue;
        if ((i + 1) % resetCounter == 0)
            expectedValue = valueOffset;
    }
}

}  // namespace smaug
