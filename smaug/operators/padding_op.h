#ifndef _OPERATORS_PADDING_OP_H_
#define _OPERATORS_PADDING_OP_H_

#include "smaug/core/backend.h"
#include "smaug/core/operator.h"
#include "smaug/core/tensor.h"
// #include "smaug/core/tensor_utils.h"
#include "smaug/core/workspace.h"

namespace smaug {

/** \ingroup Operators
 * \brief Pad a given tensor in different dimension.
 *
 * This has a software-based implementation.
 *
 * @tparam Backend The Backend that sets Alignment.
 */
template <typename Backend>
class PaddingOperator : public Operator {
   public:
    PaddingOperator(const std::string& name,
                    Workspace* workspace,
                    const std::vector<int> _padders)
            : Operator(name, OpType::Repeat, workspace), padders(_padders),
              padders(_padders) {
        inputs.resize(kNumInputs, nullptr);
        outputs.resize(kNumOutputs, nullptr);
    }

    /** Set the number of padders of the Tensor along each dimension. */
    void setPadders(const std::vector<std::vector<int>>& _padders) {
        padders = _padders;
    }

    auto getPadders() { return padders; }
    // A required function that implements the actual Operator logic.  Leave
    // this blank for now.
    // void run() override { ; }

    // Optional override for testing purposes.
    // void createAllTensors() override { ; }

    // Optional but recommended function to verify operator parameters.
    // bool validate() override { ; }
    
    enum { Inputs, kNumInputs };
    enum { Outputs, kNumOutputs };

   protected:
    std::vector<std::vector<int>> padders;
};

}  // namespace smaug

#endif
