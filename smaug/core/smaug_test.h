/**
 * \file smaug_test.h
 * \brief SMAUG unit test fixture.
 */

#include <fstream>

#include "catch.hpp"
#include "smaug/core/network.h"
#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/core/workspace.h"

namespace smaug {

/**
 * Any function that accepts a Tensor, fills it with data, and returns
 * nothing.
 */
typedef void (*FillTensorDataFunc)(Tensor* tensor);

/**
 * Sets the absolute value by which a result can differ from Approx's expected
 * value.
 */
constexpr float kMargin = 0.001;

/**
 * Set the percentage by which a result can differ from Approx's expected
 * value.
 */
constexpr float kEpsilon = 0.01;

class Operator;

/**
 * The Catch2 test fixture used by all C++ unit tests.
 *
 * This fixture encapsulates a Network and Workspace, and exposes a set of
 * useful functions for writing unit tests, like filling Tensors with random
 * data and verifying approximate equality of two Tensors.
 */
class SmaugTest {
   public:
    SmaugTest() {
        network_ = new Network("test");
        workspace_ = new Workspace();
        SmvBackend::initGlobals();
        // Set the global variables.
        runningInSimulation = false;
        useSystolicArrayWhenAvailable = false;
        numAcceleratorsAvailable = 1;
    }

    ~SmaugTest() {
        delete network_;
        delete workspace_;
        SmvBackend::freeGlobals();
    }

    /**
     * Allocates data storage for all Tensors in the Operator.
     *
     * @tparam T The data element type.
     * @param op The Operator.
     */
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

    /**
     * Fills every input Tensor in the Operator with data by calling the
     * provided FillTensorDataFunc.
     *
     * @tparam T The type of data stored in the Tensors.
     * @param op The Operator
     * @param fillTensorDataFunc A pointer to a function to auto-generate
     * testing data for the Tensor.
     */
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

    /**
     * Compares the contents of the given Tensor against an std::vector of data
     * elements and asserts (REQUIRE) that the two are approximately pointwise
     * equal. Any region of the Tensor that would be ignored by its
     * TensorIndexIterator (like alignment zero-padding) are not included in
     * the pointwise comparison.
     */
    template <typename DType>
    void verifyOutputs(Tensor* output,
                       const std::vector<DType>& expected) {
        auto ptr = output->template data<DType>();
        int i = 0;
        for (auto idx = output->startIndex(); !idx.end(); ++idx, ++i) {
            REQUIRE(Approx(ptr[idx]).margin(kMargin).epsilon(kEpsilon) ==
                    expected[i]);
        }
        REQUIRE(i == expected.size());
    }

    /**
     * Compares the contents of the two given Tensors against each other and
     * asserts (REQUIRE) that the two are approximately pointwise equal. Any
     * region of the Tensor that would be ignored by its TensorIndexIterator
     * (like alignment zero-padding) are not included in the pointwise
     * comparison.
     */
    template <typename DType>
    void verifyOutputs(Tensor* output, Tensor* expected) {
        auto outputPtr = output->template data<DType>();
        auto expectedPtr = expected->template data<DType>();
        auto outputIdx = output->startIndex();
        auto expectedIdx = expected->startIndex();
        for (; !outputIdx.end(); ++outputIdx, ++expectedIdx) {
            REQUIRE(Approx(outputPtr[outputIdx])
                            .margin(kMargin)
                            .epsilon(kEpsilon) == expectedPtr[expectedIdx]);
        }
    }

    Network* buildNetwork(const std::string& modelTopo,
                          const std::string& modelParams);
    Tensor* buildAndRunNetwork(const std::string& modelTopo,
                               const std::string& modelParams);

    Network* network() const { return network_; }
    Workspace* workspace() const { return workspace_; }

   protected:
    /**
     * Resolves a SMAUG_HOME relative path to a particular resource (like a
     * protobuf).
     */
    std::string resolvePath(const std::string& relPath) {
        const char* baseDir = std::getenv("SMAUG_HOME");
        if (baseDir == NULL)
            assert(false && "SMAUG_HOME is not set.");
        std::string fullPath = std::string(baseDir) + '/' + relPath;
        if (!std::ifstream(fullPath)) {
            std::cerr << "File " << fullPath
                      << " doesn't exist! This could be because the proto is "
                         "too large to be submit to GitHub, please check and "
                         "create it locally.\n";
            exit(0);
        }
        return fullPath;
    }

    Network* network_;
    Workspace* workspace_;
};

/** This converts a float32 into a float16. */
float16 fp16(float fp32_data);

/** This converts a float16 into a float32. */
float fp32(float16 fp16_data);

/**
 * This creates a tensor with float32 data type and fills it with data
 * converted from a source tensor with float16 data.
 */
Tensor* convertFp16ToFp32Tensor(Tensor* fp16Tensor, Workspace* workspace);

/**
 * This creates a tensor with float16 data type and fills it with data
 * converted from a source tensor with float32 data.
 */
Tensor* convertFp32ToFp16Tensor(Tensor* fp32Tensor, Workspace* workspace);

}  // namespace smaug
