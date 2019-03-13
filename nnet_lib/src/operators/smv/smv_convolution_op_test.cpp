#include "catch.hpp"
#include "core/backend.h"
#include "core/tensor.h"
#include "core/smaug_test.h"
#include "operators/smv/smv_convolution_op.h"
#include "operators/smv/smv_convolution_tiling.h"

using namespace smaug;
using namespace smaug::smv::conv;

void fillTensorWithData(Tensor* tensor) {
    const TensorShape& shape = tensor->getShape();
    // Each dimension C is initialized to a different constant value.
    float* dataPtr = tensor->data<float>();
    int resetCounter = shape.getStorageDim(3);
    int value = 0;
    for (int i = 0; i < shape.storageSize(); i++) {
        dataPtr[i] = value++;
        if ((i + 1) % resetCounter == 0)
            value = 0;
    }
}

TEST_CASE_METHOD(SmaugTest, "SMV Tiled Convolution", "[smvconv]") {
    smv::kSpadSize = 32 * 1024;
    auto convOp = new SmvConvolutionOp("conv", workspace());
    // Outputs should be the same size as inputs.
    convOp->setStride(1, 1);
    convOp->setPadding(SamePadding);

    SECTION("DimN tiled convolution") {
        TensorShape inputShape({ 1, 8, 8, 96}, DataLayout::NHWC);
        Tensor* inputs = new Tensor("inputs", inputShape);
        convOp->setInput(inputs, 0);
        convOp->setWeightDims(3, 3, 128);
        convOp->createAllTensors();
        allocateAllTensors<float>(convOp);
        fillTensorWithData(inputs);
        fillTensorWithData(convOp->getInput(1));
        convOp->run();
    }

    SECTION("DimNH tiled convolution") {
        // TODO: This will fail because the tiling logic does not account for
        // padding, so it produces three input tiles but only two output tiles.
        TensorShape inputShape({ 1, 32, 32, 16 }, DataLayout::NHWC);
        Tensor* inputs = new Tensor("inputs", inputShape);
        convOp->setInput(inputs, 0);
        convOp->setWeightDims(3, 3, 8);
        convOp->createAllTensors();
        allocateAllTensors<float>(convOp);
        fillTensorWithData(inputs);
        fillTensorWithData(convOp->getInput(1));
        convOp->run();
    }

    SECTION("DimNC tiled convolution") {
        TensorShape inputShape({ 1, 16, 8, 128 }, DataLayout::NHWC);
        Tensor* inputs =
                new Tensor("inputs", inputShape);
        convOp->setInput(inputs, 0);
        convOp->setWeightDims(3, 3, 8);
        convOp->createAllTensors();
        allocateAllTensors<float>(convOp);
        fillTensorWithData(inputs);
        fillTensorWithData(convOp->getInput(1));
        convOp->run();
    }
}
