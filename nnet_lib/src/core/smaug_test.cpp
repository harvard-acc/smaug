#include "fp16.h"
#include "core/datatypes.h"
#include "core/smaug_test.h"

namespace smaug {

// We can't directly compare with float16 values, so convert to float32.
template <>
void SmaugTest::verifyOutputs<float16>(Tensor* output, Tensor* expected) {
    auto outputPtr = output->data<float16>();
    auto expectedPtr = expected->data<float16>();
    auto outputIdx = output->startIndex();
    auto expectedIdx = expected->startIndex();
    for (; !outputIdx.end(); ++outputIdx, ++expectedIdx) {
        REQUIRE(Approx(fp32(outputPtr[outputIdx])).epsilon(0.01) ==
                fp32(expectedPtr[expectedIdx]));
    }
}

}  // namespace smaug

float16 fp16(float fp32_data) {
    return fp16_ieee_from_fp32_value(fp32_data);
}

float fp32(float16 fp16_data) {
    return fp16_ieee_to_fp32_value(fp16_data);
}

Tensor* convertFp16ToFp32Tensor(Tensor* fp16Tensor, Workspace* workspace) {
    const TensorShape& shape = fp16Tensor->getShape();
    Tensor* fp32Tensor = new Tensor(fp16Tensor->getName() + "/fp32", shape);
    fp32Tensor->allocateStorage<float>();
    workspace->addTensor(fp32Tensor);
    auto fp16DataPtr = fp16Tensor->data<float16>();
    auto fp32DataPtr = fp32Tensor->data<float>();
    auto fp16Idx = fp16Tensor->startIndex();
    auto fp32Idx = fp32Tensor->startIndex();
    for (; !fp16Idx.end(); ++fp16Idx, ++fp32Idx) {
        fp32DataPtr[fp32Idx] = fp32(fp16DataPtr[fp16Idx]);
    }
    return fp32Tensor;
}

Tensor* convertFp32ToFp16Tensor(Tensor* fp32Tensor, Workspace* workspace) {
    const TensorShape& shape = fp32Tensor->getShape();
    Tensor* fp16Tensor = new Tensor(fp32Tensor->getName() + "/fp16", shape);
    fp16Tensor->allocateStorage<float16>();
    workspace->addTensor(fp16Tensor);
    auto fp16DataPtr = fp16Tensor->data<float16>();
    auto fp32DataPtr = fp32Tensor->data<float>();
    auto fp16Idx = fp16Tensor->startIndex();
    auto fp32Idx = fp32Tensor->startIndex();
    for (; !fp16Idx.end(); ++fp16Idx, ++fp32Idx) {
        fp16DataPtr[fp16Idx] = fp16(fp32DataPtr[fp32Idx]);
    }
    return fp16Tensor;
}

