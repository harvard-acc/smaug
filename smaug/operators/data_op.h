#ifndef _OPERATORS_DATA_OP_H_
#define _OPERATORS_DATA_OP_H_

#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/core/workspace.h"

namespace smaug {

/** \ingroup Operators
 * A data operator contains a Tensor that it exposes as its only Output.
 *
 * This is the only operator that is not expected to have any inputs. Its
 * existence is to maintain the abstraction that the input to all other
 * operators is always provided by another Operator.
 *
 * @tparam Backend The Backend specialization of this Operator.
 */
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

   protected:
    Tensor* data;
};

} // namespace smaug

#endif
