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
 
/*
  void setParam1(int val) { param1 = val; }
  void setParam2(int val) { param2 = val; }

  void elementwise_add(float* input0, float* input1, float* output, int size) {
	  for (int i = 0; i < size; i++) {
		  output[i] = input0[i] + input1[i];
	  }
  }
*/
 
  // A required function that implements the actual Operator logic.  Leave this
  // blank for now.
  void run() override { }

  // Optional override for testing purposes.
  void createAllTensors() override { 
  }
 
  // Optional but recommended function to verify operator parameters.
  bool validate() override { }
 
  // An optional function to tile the input tensors.
  void tile() override {}
 
  enum {kInput0, kInput1, kNumInputs};
  enum {kOutput, kNumOutputs};
 
/*
 private:
  int param1 = 0;
  int param2 = 0;
*/
};
 
}  // namespace smaug
