#ifndef _OPERATORS_INNER_PRODUCT_OP_H_
#define _OPERATORS_INNER_PRODUCT_OP_H_

#include "core/backend.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "core/tensor_utils.h"
#include "core/workspace.h"

namespace smaug {

template <typename Backend>
class InnerProductOp : public Operator {
   public:
    InnerProductOp(const std::string& name, Workspace* workspace)
            : Operator(name, OpType::InnerProduct, workspace), numOutputs(0),
              weightsTensorsCreated(false), outputTensorsCreated(false),
              weightsName(name + "/weights") {
        inputs.resize(kNumInputs, nullptr);
        outputs.resize(kNumOutputs, nullptr);
    }

    void setNumOutputs(int _outputs) { numOutputs = _outputs; }

    virtual void run() {}
    virtual bool validate() { return numOutputs > 0 && Operator::validate(); }
    TensorShape inferOutputShape() const {
        const TensorShape& shape = getInput(Inputs)->getShape();
        assert(shape.getLayout() == DataLayout::NC);
        return TensorShape(
                { shape[0], numOutputs }, DataLayout::NC, Backend::Alignment);
    }

    TensorShape inferWeightsShape() const {
        const TensorShape& shape = getInput(Inputs)->getShape();
        assert(shape.getLayout() == DataLayout::NC);
        std::vector<int> outputDims;
        DataLayout outLayout;
        if (Backend::TransposeFCWeights) {
            outputDims = { numOutputs, shape[1] };
            outLayout = DataLayout::NC;
        } else {
            outputDims = { shape[1], numOutputs };
            outLayout = DataLayout::CN;
        }
        return TensorShape(outputDims, outLayout, Backend::Alignment);
    }

    virtual DataLayoutSet getInputDataLayouts() const {
        return DataLayoutSet(DataLayout::NC);
    }
    virtual DataLayoutSet getOutputDataLayouts() const {
        return DataLayoutSet(DataLayout::NC);
    }

    void createWeightsTensors() {
        if (inputs.at(Weights))
            return;
        TensorShape shape = inferWeightsShape();
        Tensor* weights = new Tensor(weightsName, shape);
        workspace->addTensor(weights);
        inputs.at(Weights) = weights;
        weightsTensorsCreated = true;
    }

    void createOutputTensors() {
        if (outputs.at(Outputs))
            return;
        TensorShape shape = inferOutputShape();
        Tensor* output = new Tensor(name, shape);
        workspace->addTensor(output);
        outputs[Outputs] = output;
    }

    virtual void createAllTensors() {
        createWeightsTensors();
        createOutputTensors();
    }

    int getNumOutputs() const { return numOutputs; }

    virtual void printSummary(std::ostream& out) const {
      const TensorShape& weightsShape = inputs.at(Weights)->getShape();
      const TensorShape& outputShape = outputs.at(Outputs)->getShape();
      out << this->name << " (InnerProduct)\t\t" << outputShape << "\t\t"
          << weightsShape << "\t\t" << weightsShape.size() << "\n";
    }

    virtual std::vector<TensorBase*> getParameterizableInputs() {
        return { inputs[Weights] };
    }

   public:
    enum { Inputs, Weights, kNumInputs };
    enum { Outputs, kNumOutputs };

   protected:
    int numOutputs;
    bool weightsTensorsCreated;
    bool outputTensorsCreated;
    std::string weightsName;
};

REGISTER_SPECIAL_OP(InnerProductOp, ReferenceBackend);

}  // namespace smaug

#endif
