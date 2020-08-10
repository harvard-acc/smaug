#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/greater_op.h"

#ifdef __cplusplus
extern "C" {
#endif

/** \ingroup AladdinKernels
 *
 * A Reference implementation of elementwise greater-than.
 */
void ref_greater(float* input0, float* input1, bool* results, int input_size) {
    dmaLoad(input0, input0, input_size * sizeof(float));
    dmaLoad(input1, input1, input_size * sizeof(float));
    less_loop:
    for (int i = 0; i < input_size; i++) {
        results[i] = input0[i] > input1[i];
    }
    dmaStore(results, results, input_size * sizeof(bool));
}

/** \ingroup AladdinKernels
 *
 * A Reference implementation of elementwise greater-than-or-equal-to.
 */
void ref_greater_equal(float* input0,
                       float* input1,
                       bool* results,
                       int input_size) {
    dmaLoad(input0, input0, input_size * sizeof(float));
    dmaLoad(input1, input1, input_size * sizeof(float));
    less_equal_loop:
    for (int i = 0; i < input_size; i++) {
        results[i] = input0[i] >= input1[i];
    }
    dmaStore(results, results, input_size * sizeof(bool));
}

#ifdef __cplusplus
}
#endif

namespace smaug {

template <>
void GreaterOp<ReferenceBackend>::run() {
    auto input0 = getInput(Input0);
    auto input1 = getInput(Input1);
    auto output = getOutput(Outputs);
    const TensorShape& input0Shape = input0->getShape();
    const TensorShape& input1Shape = input1->getShape();
    const TensorShape& outputShape = output->getShape();
    assert(input0Shape == input1Shape && input0Shape == outputShape);

    float* input0Data = input0->data<float>();
    float* input1Data = input1->data<float>();
    bool* outputData = output->data<bool>();
    mapArrayToAccel(ref::kEltwiseOpHw, "input0", input0Data,
                    input0Shape.storageSize() * sizeof(float));
    mapArrayToAccel(ref::kEltwiseOpHw, "input1", input1Data,
                    input1Shape.storageSize() * sizeof(float));
    mapArrayToAccel(ref::kEltwiseOpHw, "results", outputData,
                    outputShape.storageSize() * sizeof(bool));
    invokeKernel(ref::kEltwiseOpHw, ref_greater, input0Data, input1Data,
                 outputData, input0Shape.size());
}

template <>
void GreaterEqualOp<ReferenceBackend>::run() {
    auto input0 = getInput(Input0);
    auto input1 = getInput(Input1);
    auto output = getOutput(Outputs);
    const TensorShape& input0Shape = input0->getShape();
    const TensorShape& input1Shape = input1->getShape();
    const TensorShape& outputShape = output->getShape();
    assert(input0Shape == input1Shape && input0Shape == outputShape);

    float* input0Data = input0->data<float>();
    float* input1Data = input1->data<float>();
    bool* outputData = output->data<bool>();
    mapArrayToAccel(ref::kEltwiseOpHw, "input0", input0Data,
                    input0Shape.storageSize() * sizeof(float));
    mapArrayToAccel(ref::kEltwiseOpHw, "input1", input1Data,
                    input1Shape.storageSize() * sizeof(float));
    mapArrayToAccel(ref::kEltwiseOpHw, "results", outputData,
                    outputShape.storageSize() * sizeof(bool));
    invokeKernel(ref::kEltwiseOpHw, ref_greater_equal, input0Data, input1Data,
                 outputData, input0Shape.size());
}

}  // namespace smaug

