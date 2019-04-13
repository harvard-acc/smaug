#include "catch.hpp"
#include "core/backend.h"
#include "core/tensor.h"
#include "core/scheduler.h"
#include "core/smaug_test.h"
#include "operators/reorder_op.h"

using namespace smaug;

TEST_CASE_METHOD(SmaugTest, "Network tests", "[network]") {
    std::string modelPath = "../../models/";
    SECTION("ELU network. 11 layers of convolutions") {
        // ELU network with the SMV backend.
        Network* network =
                buildNetwork(modelPath + "cifar100-elu/elu_smv_topo.pbtxt",
                             modelPath + "cifar100-elu/elu_smv_params.pb");
        Tensor* output = runNetwork(network, workspace());

        // The same network with the reference backend. This is used for
        // producing expected outputs.
        Network* refNetwork =
                buildNetwork(modelPath + "cifar100-elu/elu_ref_topo.pbtxt",
                             modelPath + "cifar100-elu/elu_ref_params.pb");
        Tensor* refOutput = runNetwork(refNetwork, workspace());

        // SMV outputs need to be converted into float32 before validations.
        verifyOutputs<float>(
                convertFp16ToFp32Tensor(output, workspace()), refOutput);
    }
}
