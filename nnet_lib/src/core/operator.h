#ifndef _CORE_OPERATOR_H_
#define _CORE_OPERATOR_H_

#include <string>
#include <vector>
#include <map>

#include "core/graph.h"
#include "core/tensor.h"

namespace smaug {

class Workspace;

enum OpType {
    UnknownOp,
    Convolution3d,
    ConvolutionDepthwise,
    MaxPooling,
    AveragePooling,
    InnerProduct,
    BatchNorm,
    Data,
    ReLU,
    ELU,
    SELU,
    Tanh,
    HardTanh,
    Sigmoid,
    Softmax,
    EltwiseAdd,
    Reorder,
};

enum PaddingType {
    UnknownPadding,
    SamePadding,
    ValidPadding,
};

class Operator {
   public:
    Operator(const std::string& _name, OpType _opType, Workspace* _workspace)
            : name(_name), opType(_opType), workspace(_workspace) {}
    virtual ~Operator() {}

    virtual void run() = 0;
    virtual bool validate() = 0;
    virtual void createAllTensors() = 0;
    virtual void printSummary(std::ostream& out) const {}

    virtual DataLayoutSet getInputDataLayouts() const {
        return DataLayoutSet(DataLayout::UnknownLayout);
    }
    virtual DataLayoutSet getOutputDataLayouts() const {
        return DataLayoutSet(DataLayout::UnknownLayout);
    }

    void setInput(TensorBase* op, int index) {
        inputs[index] = op;
    }
    void setOutput(TensorBase* op, int index) {
        outputs[index] = op;
    }
    const std::string& getName() const { return name; }
    Vertex getVertex() const { return vertex; }
    void setVertex(Vertex v) { vertex = v; }
    OpType getOpType() const { return opType; }

    template <typename Backend>
    Tensor<Backend>* getInput(int index) const {
        return dynamic_cast<Tensor<Backend>*>(inputs.at(index));
    }
    TensorBase* getInput(int index) const { return inputs.at(index); }

    template <typename Backend>
    Tensor<Backend>* getOutput(int index) const {
        return dynamic_cast<Tensor<Backend>*>(outputs.at(index));
    }
    TensorBase* getOutput(int index) const { return outputs.at(index); }

   protected:
    std::vector<TensorBase*> inputs;
    std::vector<TensorBase*> outputs;
    std::string name;
    OpType opType;
    Vertex vertex;
    Workspace* workspace;
};

}  // namespace smaug

#endif
