#include "catch.hpp"
#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/core/smaug_test.h"
#include "smaug/operators/data_op.h"

using namespace smaug;

TEST_CASE_METHOD(SmaugTest, "Unpacking of float16 tensor data", "[fp16]") {
    std::string modelPath = "smaug/python/test_inputs/";
    SECTION("Unpacking tensor of [8] shape") {
        Network* network = buildNetwork(modelPath + "fp16_even_topo.txt",
                                        modelPath + "fp16_even_params.pb");
        auto dataOp = network->getOperator("input");
        auto inputTensor = dataOp->getInput(0);
        std::vector<float16> expectedValues{
            // Expected data values of 8 float16 tensors:
            // [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]
            fp16(1.1), fp16(2.2), fp16(3.3), fp16(4.4),
            fp16(5.5), fp16(6.6), fp16(7.7), fp16(8.8)
        };
        verifyOutputs(inputTensor, expectedValues);
    }

    SECTION("Unpacking tensor of [4, 3] shape") {
        Network* network = buildNetwork(modelPath + "fp16_odd_topo.txt",
                                        modelPath + "fp16_odd_params.pb");
        auto dataOp = network->getOperator("input");
        auto inputTensor = dataOp->getInput(0);
        std::vector<float16> expectedValues{
            // Expected data values of 12 float16 tensors:
            // [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11, 12.12]
            fp16(1.1), fp16(2.2),  fp16(3.3),   fp16(4.4),
            fp16(5.5), fp16(6.6),  fp16(7.7),   fp16(8.8),
            fp16(9.9), fp16(10.1), fp16(11.11), fp16(12.12)
        };
        verifyOutputs(inputTensor, expectedValues);
    }

    SECTION("Unpacking tensor of [3, 3] shape") {
        Network* network = buildNetwork(modelPath + "fp16_odd_odd_topo.txt",
                                        modelPath + "fp16_odd_odd_params.pb");
        auto dataOp = network->getOperator("input");
        auto inputTensor = dataOp->getInput(0);
        std::vector<float16> expectedValues{
            // Expected data values of 9 float16 tensors:
            // [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
            fp16(1.1), fp16(2.2), fp16(3.3), fp16(4.4), fp16(5.5),
            fp16(6.6), fp16(7.7), fp16(8.8), fp16(9.9)
        };
        verifyOutputs(inputTensor, expectedValues);
    }
}

