#include "smaug/core/backend.h"
#include "smaug/core/operator.h"
#include "smaug/core/workspace.h"
#include "smaug/operators/common.h"
 
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
  void tile() override {}
 
 
 private:
  int param1 = 0;
  int param2 = 0;
};
 
}  // namespace smaug
