#include "catch.hpp"
#include "core/network.h"
#include "core/tensor.h"
#include "core/workspace.h"

namespace smaug {

class Operator;

class SmaugTest {
   public:
    SmaugTest() {
        network_ = new Network("test");
        workspace_ = new Workspace();
    }

    ~SmaugTest() {
        delete network_;
        delete workspace_;
    }

    template <typename T, typename Backend>
    void allocateAllTensors(Operator* op) {
        for (auto t : op->getInputs()) {
            auto tensor = dynamic_cast<Tensor<Backend>*>(t);
            tensor->template allocateStorage<T>();
        }
        for (auto t : op->getOutputs()) {
            auto tensor = dynamic_cast<Tensor<Backend>*>(t);
            tensor->template allocateStorage<T>();
        }
    }

    template <typename DType, typename Backend>
    void verifyOutputs(Tensor<Backend>* output,
                       const std::vector<DType>& expected) {
        auto ptr = output->template data<DType>();
        int i = 0;
        for (auto idx = output->template startIndex(); !idx.end(); ++idx, ++i) {
            REQUIRE(Approx(ptr[idx]) == expected[i]);
        }
    }

    Network* network() const { return network_; }
    Workspace* workspace() const { return workspace_; }

   protected:
    Network* network_;
    Workspace* workspace_;
};

}  // namespace smaug
