#ifndef _CORE_OPERATOR_H_
#define _CORE_OPERATOR_H_

#include <string>
#include <vector>
#include <map>
#include <boost/format.hpp>

#include "smaug/core/typedefs.h"
#include "smaug/core/tensor.h"
#include "smaug/core/tensor_utils.h"
#include "smaug/core/types.pb.h"
#include "smaug/operators/common.h"

#define REGISTER_SPECIAL_OP(Operator, Backend)                                 \
    template <>                                                                \
    void Operator<Backend>::run();

namespace smaug {

class Workspace;

constexpr const char* kLayerFormat = "%-40s %-25s %=15d\n";

class Operator {
   public:
    Operator(const std::string& _name, OpType _opType, Workspace* _workspace)
            : name(_name), opType(_opType), workspace(_workspace),
              numPendingInputs(-1) {}
    virtual ~Operator() {}

    virtual void tile() {};
    virtual void run() = 0;
    virtual bool validate() {
        return validateInputsOutputs() && opType != OpType::UnknownOp;
    }
    virtual void createAllTensors() = 0;
    // The default implementation will just return true if any input is dead
    virtual bool isDead();
    virtual std::vector<TensorBase*> getParameterizableInputs() { return {}; }
    // This returns the number of parameterizable weights in the operator.
    virtual int getNumParameters() const { return 0; }
    virtual bool isSamplingSupported() const { return false; }
    virtual void setSamplingInfo(const SamplingInfo& sampling) {}

    void printSummary(std::ostream& out) const;
    void setInput(TensorBase* op, int index) { inputs[index] = op; }
    void setOutput(TensorBase* op, int index) { outputs[index] = op; }
    void setNumPendingInputs(int num) { numPendingInputs = num; }
    int getNumPendingInputs() const { return numPendingInputs; }
    void decrNumPendingInputs() { numPendingInputs--; }
    const std::string& getName() const { return name; }
    Vertex getVertex() const { return vertex; }
    void setVertex(Vertex v) { vertex = v; }
    OpType getOpType() const { return opType; }
    Workspace* getWorkspace() { return workspace; }

    Tensor* getInput(int index) const {
        return dynamic_cast<Tensor*>(inputs.at(index));
    }
    const std::vector<TensorBase*>& getInputs() const { return inputs; }

    Tensor* getOutput(int index) const {
        return dynamic_cast<Tensor*>(outputs.at(index));
    }
    const std::vector<TensorBase*>& getOutputs() const { return outputs; }

    void setInputsMemType(MemoryType type) { inputsMemType = type; }
    void setWeightsMemType(MemoryType type) { weightsMemType = type; }
    void setOutputsMemType(MemoryType type) { outputsMemType = type; }
    MemoryType getInputsMemType() const { return inputsMemType; }
    MemoryType getWeightsMemType() const { return weightsMemType; }
    MemoryType getOutputsMemType() const { return outputsMemType; }

   protected:
    bool tensorsAllConstructed(const std::vector<TensorBase*>& tensors) const;
    bool validateInputsOutputs() const;

    std::vector<TensorBase*> inputs;
    std::vector<TensorBase*> outputs;
    std::string name;
    OpType opType;
    Vertex vertex;
    Workspace* workspace;
    int numPendingInputs;
    // Host memory access types for different data.
    MemoryType inputsMemType;
    MemoryType weightsMemType;
    MemoryType outputsMemType;
};

}  // namespace smaug

#endif
