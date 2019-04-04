#include "catch.hpp"
#include "core/network.h"
#include "core/network_builder.h"
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

    template <typename T>
    void allocateAllTensors(Operator* op) {
        for (auto t : op->getInputs()) {
            auto tensor = dynamic_cast<Tensor*>(t);
            tensor->template allocateStorage<T>();
        }
        for (auto t : op->getOutputs()) {
            auto tensor = dynamic_cast<Tensor*>(t);
            tensor->template allocateStorage<T>();
        }
    }

    template <typename DType>
    void verifyOutputs(Tensor* output,
                       const std::vector<DType>& expected) {
        auto ptr = output->template data<DType>();
        int i = 0;
        for (auto idx = output->startIndex(); !idx.end(); ++idx, ++i) {
            REQUIRE(Approx(ptr[idx]) == expected[i]);
        }
        REQUIRE(i == expected.size());
    }

    template <typename DType>
    void verifyOutputs(Tensor* output, Tensor* expected) {
        auto outputPtr = output->template data<DType>();
        auto expectedPtr = expected->template data<DType>();
        auto outputIdx = output->startIndex();
        auto expectedIdx = expected->startIndex();
        for (; !outputIdx.end(); ++outputIdx, ++expectedIdx) {
            REQUIRE(Approx(outputPtr[outputIdx]) == expectedPtr[expectedIdx]);
        }
    }

    Network* buildNetwork(const std::string& modelpb) {
        if (network_ != nullptr)
            delete network_;
        network_ = smaug::buildNetwork(resolvePath(modelpb), workspace_);
        return network_;
    }

    Network* network() const { return network_; }
    Workspace* workspace() const { return workspace_; }

   protected:
    std::string resolvePath(const std::string& relPath) {
        const char* baseDir = std::getenv("SMAUG_HOME");
        if (baseDir == NULL)
            assert(false && "SMAUG_HOME is not set.");
        return std::string(baseDir) + "/nnet_lib/src/" + relPath;
    }

    Network* network_;
    Workspace* workspace_;
};

}  // namespace smaug
