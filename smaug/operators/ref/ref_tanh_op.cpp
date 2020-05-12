#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/tanh_op.h"
#include "smaug/operators/ref/ref_activation_fun_op.h"

namespace smaug {

template <>
void TanhOp<ReferenceBackend>::run() {
    auto inputs = getInput(Inputs);
    auto outputs = getOutput(Outputs);
    assert(inputs->getShape() == outputs->getShape());
    float* inputData = inputs->data<float>();
    float* outputData = outputs->data<float>();
    mapArrayToAccel(ref::kEltwiseOpHw, "inputs", inputData,
                    inputs->getShape().storageSize() * sizeof(float));
    mapArrayToAccel(ref::kEltwiseOpHw, "results", outputData,
                    inputs->getShape().storageSize() * sizeof(float));
    activation_type function = activation_type::TANH;
    activation_param_t params;
    invokeKernel(ref::kEltwiseOpHw, ref_activation_fun_nc, inputData,
                 outputData, inputs->getShape().size(), function, params);
}

template <>
void HardTanhOp<ReferenceBackend>::run() {
    auto inputs = getInput(Inputs);
    auto outputs = getOutput(Outputs);
    assert(inputs->getShape() == outputs->getShape());
    float* inputData = inputs->data<float>();
    float* outputData = outputs->data<float>();
    mapArrayToAccel(ref::kEltwiseOpHw, "inputs", inputData,
                    inputs->getShape().storageSize() * sizeof(float));
    mapArrayToAccel(ref::kEltwiseOpHw, "results", outputData,
                    inputs->getShape().storageSize() * sizeof(float));
    activation_type function = activation_type::HARD_TANH;
    activation_param_t params;
    params.min = min;
    params.max = max;
    invokeKernel(ref::kEltwiseOpHw, ref_activation_fun_nc, inputData,
                 outputData, inputs->getShape().size(), function, params);
}

}  // namespace smaug
