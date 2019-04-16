#include "catch.hpp"
#include "core/network.h"
#include "core/network_builder.h"
#include "core/backend.h"
#include "core/tensor.h"
#include "core/workspace.h"

using namespace smaug;

typedef void (*FillTensorDataFunc)(Tensor* tensor);

namespace smaug {

class Operator;

class SmaugTest {
   public:
    SmaugTest() {
        network_ = new Network("test");
        workspace_ = new Workspace();
        SmvBackend::initGlobals();
    }

    ~SmaugTest() {
        delete network_;
        delete workspace_;
        SmvBackend::freeGlobals();
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

    template <typename T>
    void createAndFillTensorsWithData(Operator* op,
                                      FillTensorDataFunc fillTensorDataFunc) {
        op->createAllTensors();
        allocateAllTensors<T>(op);
        for (auto input : op->getInputs()) {
            Tensor* tensor = dynamic_cast<Tensor*>(input);
            fillTensorDataFunc(tensor);
        }
    }

    template <typename DType>
    void verifyOutputs(Tensor* output,
                       const std::vector<DType>& expected) {
        auto ptr = output->template data<DType>();
        int i = 0;
        for (auto idx = output->startIndex(); !idx.end(); ++idx, ++i) {
            REQUIRE(Approx(ptr[idx]).epsilon(0.01) == expected[i]);
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
            REQUIRE(Approx(outputPtr[outputIdx]).epsilon(0.01) ==
                    expectedPtr[expectedIdx]);
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

// This converts a float32 into a float16.
float16 fp16(float fp32_data);

// This converts a float16 into a float32.
float fp32(float16 fp16_data);

// This creates a tensor with float32 data type and fills it with data converted
// from a source tensor with float16 data.
Tensor* convertFp16ToFp32Tensor(Tensor* fp16Tensor, Workspace* workspace);

// This creates a tensor with float16 data type and fills it with data converted
// from a source tensor with float32 data.
Tensor* convertFp32ToFp16Tensor(Tensor* fp32Tensor, Workspace* workspace);
