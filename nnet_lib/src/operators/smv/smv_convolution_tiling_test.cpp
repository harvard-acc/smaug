#include "catch.hpp"
#include "core/backend.h"
#include "core/tensor.h"
#include "core/smaug_test.h"
#include "operators/smv/smv_convolution_op.h"
#include "operators/smv/smv_convolution_tiling.h"

using namespace smaug;
using namespace smaug::smv::conv;

TEST_CASE_METHOD(SmaugTest, "Basic tiling tests", "[smvtiling]") {
    smv::kSpadSize = 32 * 1024;
    auto convOp = new SmvConvolutionOp("conv", workspace());
    // Outputs should be the same size as inputs.
    convOp->setStride(1, 1);
    convOp->setPadding(SamePadding);

    SECTION("No tiling needed") {
        TensorShape inputShape({ 1, 32, 32, 8 }, DataLayout::NHWC);
        Tensor<SmvBackend>* inputs =
                new Tensor<SmvBackend>("inputs", inputShape);
        convOp->setInput(inputs, 0);
        convOp->setWeightDims(5, 5, 8);
        convOp->createAllTensors();
        allocateAllTensors<float, SmvBackend>(convOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(convOp);
        REQUIRE(config.inputs == inputShape);
        REQUIRE(config.weights.dims() == std::vector<int>{ 8, 5, 5, 8 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 32, 32, 8 });
    }

    SECTION("DimNH tiling on inputs when less than 32 channels") {
        TensorShape inputShape({ 1, 32, 32, 16 }, DataLayout::NHWC);
        Tensor<SmvBackend>* inputs =
                new Tensor<SmvBackend>("inputs", inputShape);
        TensorShape weightShape({ 8, 3, 3, 16 }, DataLayout::NHWC);
        convOp->setInput(inputs, 0);
        convOp->setWeightDims(3, 3, 8);
        convOp->createAllTensors();
        allocateAllTensors<float, SmvBackend>(convOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(convOp);
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 16, 32, 16 });
        REQUIRE(config.weights == weightShape);
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 16, 32, 8 });
    }

    SECTION("DimNC tiling on inputs and weights with > 32 channels") {
        // Pick 16x8 plane size so that there are more possibilities for channel
        // sizes than just one.
        TensorShape inputShape({ 1, 16, 8, 128 }, DataLayout::NHWC);
        Tensor<SmvBackend>* inputs =
                new Tensor<SmvBackend>("inputs", inputShape);
        convOp->setInput(inputs, 0);
        convOp->setWeightDims(3, 3, 8);
        convOp->createAllTensors();
        allocateAllTensors<float, SmvBackend>(convOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(convOp);
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 16, 8, 64 });
        REQUIRE(config.weights.dims() == std::vector<int>{ 8, 3, 3, 64 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 16, 8, 8 });
    }

    SECTION("DimN tiling for weights, None for inputs") {
        // Inputs can all fit.
        TensorShape inputShape({ 1, 8, 8, 96}, DataLayout::NHWC);
        Tensor<SmvBackend>* inputs =
                new Tensor<SmvBackend>("inputs", inputShape);
        convOp->setInput(inputs, 0);
        // ...but weights can't. The scratchpad can fit up to 9 complete
        // filters at once, so we just need DimN tiling.
        convOp->setWeightDims(3, 3, 128);
        convOp->createAllTensors();
        allocateAllTensors<float, SmvBackend>(convOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(convOp);
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 8, 8, 96 });
        REQUIRE(config.weights.dims() == std::vector<int>{ 8, 3, 3, 96 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 8, 8, 8 });
    }

    SECTION("DimNC tiling for weights, None for inputs") {
        // Inputs can all fit.
        TensorShape inputShape({ 1, 8, 8, 128}, DataLayout::NHWC);
        Tensor<SmvBackend>* inputs =
                new Tensor<SmvBackend>("inputs", inputShape);
        convOp->setInput(inputs, 0);
        // ...but weights can't. The scratchpad can only fit up to 7 complete
        // filters at once, which is smaller than the minimum 8 required for
        // DimN tiling, so we'll need to go to DimNC.
        convOp->setWeightDims(3, 3, 128);
        convOp->createAllTensors();
        allocateAllTensors<float, SmvBackend>(convOp);
        TilingConfig config = TilingOptimizer::computeBasicTileShapes(convOp);
        // Inputs don't need to be tiled.
        REQUIRE(config.inputs.dims() == std::vector<int>{ 1, 8, 8, 128 });
        // In terms of weight utilization, 24x3x3x32 is the same as 8x3x3x96,
        // but it allows us to store up to 24 partial output feature maps
        // instead of just 8, so the overall utilization of the SRAMs is
        // higher.
        REQUIRE(config.weights.dims() == std::vector<int>{ 24, 3, 3, 32 });
        REQUIRE(config.outputs.dims() == std::vector<int>{ 1, 8, 8, 24 });
    }
}
