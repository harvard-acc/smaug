#ifndef _OPERATORS_DATA_OP_H_
#define _OPERATORS_DATA_OP_H_

#include "core/tensor.h"
#include "core/workspace.h"

namespace smaug {

template <typename Backend>
class DataOp : public Operator {
   public:
    DataOp(const std::string& name, Workspace* workspace)
            : Operator(name, OpType::Data, workspace), data(NULL) {
        inputs.resize(1, nullptr);
        outputs.resize(1, nullptr);
    }

    void setData(Tensor<Backend>* _data) {
        data = _data;
        inputs[0] = data;
        outputs[0] = data;
    }

    virtual void run() {}
    virtual bool validate() { return data != NULL; }
    virtual void createAllTensors() {}

    virtual DataLayoutSet getInputDataLayouts() const {
        return DataLayoutSet(data->getShape().getLayout());
    }
    virtual DataLayoutSet getOutputDataLayouts() const {
        return getInputDataLayouts();
    }
    virtual void printSummary(std::ostream& out) const {
        const TensorShape& shape = data->getShape();
        out << name << " (Data)\t\t\t" << shape << "\n";
    }

   protected:
    Tensor<Backend>* data;
};

} // namespace smaug
#endif
