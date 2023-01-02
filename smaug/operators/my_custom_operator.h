#include "smaug/core/backend.h"
#include "smaug/core/operator.h"
#include "smaug/core/workspace.h"
#include "smaug/operators/common.h"
#include "smaug/core/tensor.h"
#include "smaug/core/tensor_utils.h"
 
namespace smaug {
 
template <typename Backend>
class MyCustomOperator : public Operator {
 public:
  MyCustomOperator(const std::string& name, Workspace* workspace) :
    Operator(name, OpType::MyCustom, workspace) {
      inputs.resize(kNumInputs, nullptr);
      outputs.resize(kNumOutputs, nullptr);
  }
 
  enum {kInput0, kInput1, kNumInputs};
  enum {kOutput, kNumOutputs};

  void setParam1(int val) { param1 = val; }
  void setParam2(int val) { param2 = val; }

  void elementwise_add(float* input0, float* input1, float* output, int size) {
	  for (int i = 0; i < size; i++) {
		  output[i] = input0[i] + input1[i];
	  }
  }
/*
*/
 
  // A required function that implements the actual Operator logic.  Leave this
  // blank for now.
  void run() override { 
  	Tensor* input0 = getInput(kInput0);
  	Tensor* input1 = getInput(kInput1);
  	Tensor* output = getOutput(kInput0);
 
  	// Get handles to the actual underlying data storage. This performs a
  	// dynamic_cast to the specified data type, which we verified is safe inside
  	// validate().
  	float* input0Data = input0->data<float>();
  	float* input1Data = input1->data<float>();
  	float* outputData = output->data<float>();
 
  	elementwise_add(input0Data, input1Data, outputData, output->getShape().size());
  }

  void run_tiled() {
	TiledTensor& input0 = tiledTensors[kInput0];
  	TiledTensor& input1 = tiledTensors[kInput1];
  	TiledTensor& output = tiledTensors[kOutput];

  	Tensor* tensorOutput = getOutput(kInput0);
 
  	for (int i = 0; i < input0.size(); i++) {
  	  Tensor* input0Tile = input0.getTileWithData(i);
  	  Tensor* input1Tile = input1.getTileWithData(i);
  	  Tensor* outputTile = output.getTileWithData(i);
 
  	  // Get handles to the actual underlying data storage. This performs a
  	  // dynamic_cast to the specified data type, which we verified is safe inside
  	  // validate().
  	  float* input0Data = input0Tile->data<float>();
  	  float* input1Data = input1Tile->data<float>();
  	  float* outputData = outputTile->data<float>();
  	  elementwise_add(input0Data, input1Data, outputData, outputTile->getShape().size());
  	}
  	// The results of the elementwise_add are stored in the tiled tensor. We need
  	// to merge the data from the individual tiles back into a single contiguous
  	// Tensor.
  	flattenTiledTensor(tiledTensors[kOutput], tensorOutput);
  }

  // Optional override for testing purposes.
  void createAllTensors() override { 
  	Tensor* output = new Tensor(name, inputs.at(kInput0)->getShape());
	outputs.at(kOutput) = output;
  	workspace->addTensor(output);
  }
 
  // Optional but recommended function to verify operator parameters.
  bool validate() override { 
  	Tensor* input0 = getInput(kInput0);
  	Tensor* input1 = getInput(kInput1);
  	return (input0->getShape() == input1->getShape() ||
  	        input0->getDataType() != DataType::Float32 ||
  	        input1->getDataType() != DataType::Float32);
  }
 
  // An optional function to tile the input tensors.
  void tile() override {
    auto inputs0 = getInput(kInput0);
    auto inputs1 = getInput(kInput1);
    auto outputs = getOutput(kOutput);
    // The simplest tiling strategy is to tile per batch. Each tile will have a
    // size of at most 1 x maxTileSize.
    int maxTileSize =
            std::min(ReferenceBackend::SpadSize() / inputs0->getDataTypeSize(),
                      inputs0->getShape().storageSize());
    TensorShape tileShape(
             { 1, maxTileSize }, DataLayout::NC, ReferenceBackend::Alignment);
    // The final bool parameter specifies whether to copy the data from the
    // source tensor into each of its tiles. Obivously, we want to do this for the
    // input tensors, but the output tensor is empty, so there's no need to
    // waste time on that.
    tiledTensors[0] = generateTiledTensorPerBatchNC(inputs0, tileShape, this, false);
    tiledTensors[1] = generateTiledTensorPerBatchNC(inputs1, tileShape, this, false);
    tiledTensors[2] = generateTiledTensorPerBatchNC(outputs, tileShape, this, false);
  }

 
 
 private:
  int param1 = 0;
  int param2 = 0;
  // Because tensor tiling is done at the start of the program (before the
  // operator starts running), these tiles need to be stored in memory for use
  // later.
  std::array<TiledTensor, 3> tiledTensors;
};
 
}  // namespace smaug
