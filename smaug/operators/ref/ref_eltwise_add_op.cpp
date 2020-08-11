#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/eltwise_add_op.h"

#ifdef __cplusplus
extern "C" {
#endif

/** \ingroup AladdinKernels
 *
 * A Reference implementation of elementwise addition.
 */
void ref_eltwise_add(float* input0,
                     float* input1,
                     float* results,
                     int input_size) {
    dmaLoad(input0, input0, input_size * sizeof(float));
    dmaLoad(input1, input1, input_size * sizeof(float));
    eltwise_add_loop:
    for (int i = 0; i < input_size; i++) {
        results[i] = input0[i] + input1[i];
    }
    dmaStore(results, results, input_size * sizeof(float));
}

#ifdef __cplusplus
}
#endif

namespace smaug {

template <>
void EltwiseAddOp<ReferenceBackend>::run() {
    auto input0 = getInput(Input0);
    auto input1 = getInput(Input1);
    auto output = getOutput(Outputs);
    const TensorShape& input0Shape = input0->getShape();
    const TensorShape& input1Shape = input1->getShape();
    const TensorShape& outputShape = output->getShape();
    assert(input0Shape == input1Shape && input0Shape == outputShape);

    float* input0Data = input0->data<float>();
    float* input1Data = input1->data<float>();
    float* outputData = output->data<float>();
    mapArrayToAccel(ref::kEltwiseOpHw, "input0", input0Data,
                    input0Shape.storageSize() * sizeof(float));
    mapArrayToAccel(ref::kEltwiseOpHw, "input1", input1Data,
                    input1Shape.storageSize() * sizeof(float));
    mapArrayToAccel(ref::kEltwiseOpHw, "results", outputData,
                    outputShape.storageSize() * sizeof(float));
    invokeKernel(ref::kEltwiseOpHw, ref_eltwise_add, input0Data, input1Data,
                 outputData, input0Shape.size());
}

}  // namespace smaug

