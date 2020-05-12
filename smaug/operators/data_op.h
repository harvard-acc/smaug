#ifndef _OPERATORS_DATA_OP_H_
#define _OPERATORS_DATA_OP_H_

#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/core/tensor_utils.h"
#include "smaug/core/workspace.h"

namespace smaug {

template <typename Backend>
class DataOp : public Operator {
   public:
    DataOp(const std::string& name, Workspace* workspace)
            : Operator(name, OpType::Data, workspace), data(NULL) {
        inputs.resize(1, nullptr);
        outputs.resize(1, nullptr);
    }

    void setData(Tensor* _data) {
        data = _data;
        inputs[0] = data;
        outputs[0] = data;
    }

    void run() override {}
    bool validate() override { return data != NULL && Operator::validate(); }
    void createAllTensors() override {}

    void printSummary(std::ostream& out) const override {
        const TensorShape& shape = data->getShape();
        out << name << " (Data)\t\t\t" << shape << "\n";
    }

   protected:
    Tensor* data;
};

} // namespace smaug

#endif
