#include "catch.hpp"
#include "smaug/core/backend.h"
#include "smaug/core/smaug_test.h"
#include "smaug/core/tensor.h"
#include "smaug/operators/padding_op.h"

using namespace smaug;

TEST_CASE_METHOD(SmaugTest, "padding a tensor", "[refop]") {
    SECTION("4D zero padding") {
        TensorShape inputShape({ 1, 1, 1, 1 }, DataLayout::NCHW);
        Tensor* input = new Tensor("input", inputShape);
        input->allocateStorage<float>();
        std::vector<float> inputValues{
            0,  // input 0, chan 0, row 0
        };
        input->fillData(inputValues.data(), inputValues.size());
        workspace()->addTensor(input);

        // Create the operator and fill it with our tensors.
        auto paddingOp =
                new PaddingOp<ReferenceBackend>("padding", workspace());
        paddingOp->setInput(input, 0);
        paddingOp->setPaddingSize({ 0, 0, 0, 0, 0, 0, 0, 0 });
        paddingOp->createAllTensors();
        // Allocates memory for all the output tensors created by
        // createAllTensors.
        allocateAllTensors<float>(paddingOp);

        paddingOp->run();
        auto output = paddingOp->getOutput(0);
        // Compare the output of the operator against expected values.
        std::vector<float> expected_output{
            0,
        };
        // This performs an approximate comparison between the tensor's output
        // and the expected values.
        REQUIRE(output->getShape().dims() == std::vector<int>{ 1, 1, 1, 1 });
        verifyOutputs(output, expected_output);
    }

    SECTION("2D 1 value") {
        TensorShape inputShape({ 1, 1 }, DataLayout::NC);
        Tensor* input = new Tensor("input", inputShape);
        input->allocateStorage<float>();
        std::vector<float> inputValues{
            1,  // input 0, chan 0
        };
        input->fillData(inputValues.data(), inputValues.size());
        workspace()->addTensor(input);

        // Create the operator and fill it with our tensors.
        auto paddingOp =
                new PaddingOp<ReferenceBackend>("padding", workspace());
        paddingOp->setInput(input, 0);
        paddingOp->setPaddingSize({ 0, 0, 1, 1 });
        paddingOp->createAllTensors();
        // Allocates memory for all the output tensors created by
        // createAllTensors.
        allocateAllTensors<float>(paddingOp);

        paddingOp->run();
        auto output = paddingOp->getOutput(0);
        // Compare the output of the operator against expected values.
        std::vector<float> expected_output{
            0,  // input 0, chan -1
            1,  // input 0, chan 0
            0,  // input 0, chan 1
        };
        // This performs an approximate comparison between the tensor's output
        // and the expected values.
        REQUIRE(output->getShape().dims() == std::vector<int>{ 1, 3 });
        verifyOutputs(output, expected_output);
    }

    SECTION("4D 1 value") {
        TensorShape inputShape({ 1, 1, 1, 1 }, DataLayout::NCHW);
        Tensor* input = new Tensor("input", inputShape);
        input->allocateStorage<float>();
        std::vector<float> inputValues{
            0,  // input 0, chan 0, row 0
        };
        input->fillData(inputValues.data(), inputValues.size());
        workspace()->addTensor(input);

        // Create the operator and fill it with our tensors.
        auto paddingOp =
                new PaddingOp<ReferenceBackend>("padding", workspace());
        paddingOp->setInput(input, 0);
        paddingOp->setPaddingSize({ 0, 0, 0, 0, 1, 1, 1, 1 });
        paddingOp->createAllTensors();
        // Allocates memory for all the output tensors created by
        // createAllTensors.
        allocateAllTensors<float>(paddingOp);

        paddingOp->run();
        auto output = paddingOp->getOutput(0);
        // Compare the output of the operator against expected values.
        std::vector<float> expected_output{
            0, 0, 0,  // input 0, chan 0, row -1
            0, 0, 0,  // input 0, chan 0, row 0
            0, 0, 0,  // input 0, chan 1, row 1
        };
        // This performs an approximate comparison between the tensor's output
        // and the expected values.
        REQUIRE(output->getShape().dims() == std::vector<int>{ 1, 1, 3, 3 });
        verifyOutputs(output, expected_output);
    }

    SECTION("4D multiple values") {
        TensorShape inputShape({ 1, 2, 3, 4 }, DataLayout::NCHW);
        Tensor* input = new Tensor("input", inputShape);
        input->allocateStorage<float>();
        std::vector<float> inputValues{
            1,  2,   3,   4,   // input 0, chan 0, row 0
            5,  6,   7,   8,   // input 0, chan 0, row 1
            9,  10,  11,  12,  // input 0, chan 0, row 2
            -1, -2,  -3,  -4,  // input 0, chan 1, row 0
            -5, -6,  -7,  -8,  // input 0, chan 1, row 1
            -9, -10, -11, -12  // input 0, chan 1, row 2
        };
        input->fillData(inputValues.data(), inputValues.size());
        workspace()->addTensor(input);

        // Create the operator and fill it with our tensors.
        auto paddingOp =
                new PaddingOp<ReferenceBackend>("padding", workspace());
        paddingOp->setInput(input, 0);
        paddingOp->setPaddingSize({ 0, 0, 0, 0, 1, 1, 1, 1 });
        paddingOp->createAllTensors();
        // Allocates memory for all the output tensors created by
        // createAllTensors.
        allocateAllTensors<float>(paddingOp);

        paddingOp->run();
        auto output = paddingOp->getOutput(0);
        // Compare the output of the operator against expected values.
        std::vector<float> expected_output{
            0, 0,  0,   0,   0,   0,  // input 0, chan 0, row -1
            0, 1,  2,   3,   4,   0,  // input 0, chan 0, row 0
            0, 5,  6,   7,   8,   0,  // input 0, chan 0, row 1
            0, 9,  10,  11,  12,  0,  // input 0, chan 0, row 2
            0, 0,  0,   0,   0,   0,  // input 0, chan 0, row 3
            0, 0,  0,   0,   0,   0,  // input 0, chan 0, row -1
            0, -1, -2,  -3,  -4,  0,  // input 0, chan 1, row 0
            0, -5, -6,  -7,  -8,  0,  // input 0, chan 1, row 1
            0, -9, -10, -11, -12, 0,  // input 0, chan 1, row 2
            0, 0,  0,   0,   0,   0,  // input 0, chan 1, row 3
        };
        // This performs an approximate comparison between the tensor's output
        // and the expected values.
        REQUIRE(output->getShape().dims() == std::vector<int>{ 1, 2, 5, 6 });
        verifyOutputs(output, expected_output);
    }

    SECTION("4D multiple values asymmetric padding") {
        TensorShape inputShape({ 1, 2, 3, 4 }, DataLayout::NCHW);
        Tensor* input = new Tensor("input", inputShape);
        input->allocateStorage<float>();
        std::vector<float> inputValues{
            1,  2,   3,   4,   // input 0, chan 0, row 0
            5,  6,   7,   8,   // input 0, chan 0, row 1
            9,  10,  11,  12,  // input 0, chan 0, row 2
            -1, -2,  -3,  -4,  // input 0, chan 1, row 0
            -5, -6,  -7,  -8,  // input 0, chan 1, row 1
            -9, -10, -11, -12  // input 0, chan 1, row 2
        };
        input->fillData(inputValues.data(), inputValues.size());
        workspace()->addTensor(input);

        // Create the operator and fill it with our tensors.
        auto paddingOp =
                new PaddingOp<ReferenceBackend>("padding", workspace());
        paddingOp->setInput(input, 0);
        paddingOp->setPaddingSize({ 0, 0, 0, 0, 1, 1, 1, 2 });
        paddingOp->createAllTensors();
        // Allocates memory for all the output tensors created by
        // createAllTensors.
        allocateAllTensors<float>(paddingOp);

        paddingOp->run();
        auto output = paddingOp->getOutput(0);
        // Compare the output of the operator against expected values.
        std::vector<float> expected_output{
            0, 0,  0,   0,   0,   0, 0,  // input 0, chan 0, row -1
            0, 1,  2,   3,   4,   0, 0,  // input 0, chan 0, row 0
            0, 5,  6,   7,   8,   0, 0,  // input 0, chan 0, row 1
            0, 9,  10,  11,  12,  0, 0,  // input 0, chan 0, row 2
            0, 0,  0,   0,   0,   0, 0,  // input 0, chan 0, row 3
            0, 0,  0,   0,   0,   0, 0,  // input 0, chan 0, row -1
            0, -1, -2,  -3,  -4,  0, 0,  // input 0, chan 1, row 0
            0, -5, -6,  -7,  -8,  0, 0,  // input 0, chan 1, row 1
            0, -9, -10, -11, -12, 0, 0,  // input 0, chan 1, row 2
            0, 0,  0,   0,   0,   0, 0,  // input 0, chan 1, row 3
        };
        // This performs an approximate comparison between the tensor's output
        // and the expected values.
        REQUIRE(output->getShape().dims() == std::vector<int>{ 1, 2, 5, 7 });
        verifyOutputs(output, expected_output);
    }
}