#include "catch.hpp"
#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/core/smaug_test.h"
#include "smaug/operators/reshape_op.h"

using namespace smaug;

TEST_CASE_METHOD(SmaugTest, "Reshape reference backend", "[refop]") {
    auto reshapeOp = new ReshapeOp<ReferenceBackend>("reshape", workspace());
    TensorShape inputShape(
            { 2, 4 }, DataLayout::NC, ReferenceBackend::Alignment);
    Tensor* input = new Tensor("input", inputShape);
    input->allocateStorage<float>();
    // The alignment for Reference is 0.
    std::vector<float> inputValues{
        1, 2, 3, 4,  // input 0
        5, 6, 7, 8   // input 1
    };
    input->fillData(inputValues.data(), inputValues.size());
    workspace()->addTensor(input);
    reshapeOp->setInput(input, 0);

    SECTION("Reshape without changing number of dims") {
        reshapeOp->setShape({ 4, 2 }, DataLayout::NC);
        reshapeOp->createAllTensors();
        allocateAllTensors<float>(reshapeOp);
        reshapeOp->run();
        auto output = reshapeOp->getOutput(0);
        REQUIRE(output->getShape().getLayout() == DataLayout::NC);
        REQUIRE(output->getShape().dims() == std::vector<int>{ 4, 2 });
        verifyOutputs(output, inputValues);
    }

    SECTION("Reshape NC to NHWC") {
        reshapeOp->setShape({ 1, 2, 2, 2 }, DataLayout::NHWC);
        reshapeOp->createAllTensors();
        allocateAllTensors<float>(reshapeOp);
        reshapeOp->run();
        auto output = reshapeOp->getOutput(0);
        REQUIRE(output->getShape().getLayout() == DataLayout::NHWC);
        REQUIRE(output->getShape().dims() == std::vector<int>{ 1, 2, 2, 2 });
        verifyOutputs(output, inputValues);
    }
}

TEST_CASE_METHOD(SmaugTest, "Reshape SMV backend", "[smvop]") {
    auto reshapeOp = new ReshapeOp<SmvBackend>("reshape", workspace());
    TensorShape inputShape({ 2, 4 }, DataLayout::NC, SmvBackend::Alignment);
    Tensor* input32 = new Tensor("input_f32", inputShape);
    input32->allocateStorage<float>();
    // The alignment for SMV is 8
    std::vector<float> inputValues{
        1, 2, 3, 4, 0, 0, 0, 0,  // input 0
        5, 6, 7, 8, 0, 0, 0, 0   // input 1
    };
    input32->fillData(inputValues.data(), inputValues.size());
    workspace()->addTensor(input32);
    Tensor* input = convertFp32ToFp16Tensor(input32, workspace());
    reshapeOp->setInput(input, 0);

    SECTION("Reshape with different paddings") {
        reshapeOp->setShape({ 4, 2 }, DataLayout::NC);
        reshapeOp->createAllTensors();
        allocateAllTensors<float16>(reshapeOp);
        reshapeOp->run();
        auto output = reshapeOp->getOutput(0);
        REQUIRE(output->getShape().getLayout() == DataLayout::NC);
        REQUIRE(output->getShape().dims() == std::vector<int>{ 4, 2 });
        verifyOutputs<float16>(output, input);
    }

    SECTION("New shape that gets rid of paddings") {
        reshapeOp->setShape({ 1, 8 }, DataLayout::NC);
        reshapeOp->createAllTensors();
        allocateAllTensors<float16>(reshapeOp);
        reshapeOp->run();
        auto output = reshapeOp->getOutput(0);
        REQUIRE(output->getShape().getLayout() == DataLayout::NC);
        REQUIRE(output->getShape().dims() == std::vector<int>{ 1, 8 });
        // The paddings are gone.
        REQUIRE(output->getShape().getPadding(1) == 0);
        verifyOutputs<float16>(output, input);
    }

    SECTION("Reshape NC to NHWC") {
        reshapeOp->setShape({ 1, 2, 2, 2 }, DataLayout::NHWC);
        reshapeOp->createAllTensors();
        allocateAllTensors<float16>(reshapeOp);
        reshapeOp->run();
        auto output = reshapeOp->getOutput(0);
        REQUIRE(output->getShape().getLayout() == DataLayout::NHWC);
        REQUIRE(output->getShape().dims() == std::vector<int>{ 1, 2, 2, 2 });
        verifyOutputs<float16>(output, input);
    }
}
