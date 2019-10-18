#include "catch.hpp"
#include "core/backend.h"
#include "core/tensor.h"
#include "core/scheduler.h"
#include "core/smaug_test.h"
#include "operators/reorder_op.h"

using namespace smaug;

TEST_CASE_METHOD(SmaugTest, "Network tests", "[network]") {
    std::string modelPath = "../../experiments/models/";
    SECTION("Minerva network. 4 layers of FCs.") {
        // Minerva network with the SMV backend.
        Network* network =
                buildNetwork(modelPath + "minerva/minerva_smv_topo.pbtxt",
                             modelPath + "minerva/minerva_smv_params.pb");
        Tensor* output = runNetwork(network, workspace());

        // The same network with the reference backend. This is used for
        // producing expected outputs.
        Network* refNetwork =
                buildNetwork(modelPath + "minerva/minerva_ref_topo.pbtxt",
                             modelPath + "minerva/minerva_ref_params.pb");
        Tensor* refOutput = runNetwork(refNetwork, workspace());

        // SMV outputs need to be converted into float32 before validations.
        verifyOutputs<float>(
                convertFp16ToFp32Tensor(output, workspace()), refOutput);
    }

    SECTION("LeNet5 network. 2 layers of Convs, 1 layout of Pool and 2 layers "
            "of FCs.") {
        // LeNet5 network with the SMV backend.
        Network* network =
                buildNetwork(modelPath + "lenet5/lenet5_smv_topo.pbtxt",
                             modelPath + "lenet5/lenet5_smv_params.pb");
        Tensor* output = runNetwork(network, workspace());

        // The same network with the reference backend. This is used for
        // producing expected outputs.
        Network* refNetwork =
                buildNetwork(modelPath + "lenet5/lenet5_ref_topo.pbtxt",
                             modelPath + "lenet5/lenet5_ref_params.pb");
        Tensor* refOutput = runNetwork(refNetwork, workspace());

        // SMV outputs need to be converted into float32 before validations.
        verifyOutputs<float>(
                convertFp16ToFp32Tensor(output, workspace()), refOutput);
    }

    SECTION("ELU network. 11 layers of convolutions and 5 layers of "
            "poolings.") {
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

    SECTION("CIFAR-10 VGG network. 10 Convs, 4 Pools and 2 FCs.") {
        // VGG network with the SMV backend.
        Network* network =
                buildNetwork(modelPath + "cifar10-vgg/vgg_smv_topo.pbtxt",
                             modelPath + "cifar10-vgg/vgg_smv_params.pb");
        Tensor* output = runNetwork(network, workspace());

        // The same network with the reference backend. This is used for
        // producing expected outputs.
        Network* refNetwork =
                buildNetwork(modelPath + "cifar10-vgg/vgg_ref_topo.pbtxt",
                             modelPath + "cifar10-vgg/vgg_ref_params.pb");
        Tensor* refOutput = runNetwork(refNetwork, workspace());

        // SMV outputs need to be converted into float32 before validations.
        verifyOutputs<float>(
                convertFp16ToFp32Tensor(output, workspace()), refOutput);
    }

    SECTION("ELU large network. 19 layers of convolutions and 5 layers of "
            "poolings.") {
        // ELU network with the SMV backend.
        Network* network = buildNetwork(
                modelPath + "cifar100-large-elu/large_elu_smv_topo.pbtxt",
                modelPath + "cifar100-large-elu/large_elu_smv_params.pb");
        Tensor* output = runNetwork(network, workspace());

        // The same network with the reference backend. This is used for
        // producing expected outputs.
        Network* refNetwork = buildNetwork(
                modelPath + "cifar100-large-elu/large_elu_ref_topo.pbtxt",
                modelPath + "cifar100-large-elu/large_elu_ref_params.pb");
        Tensor* refOutput = runNetwork(refNetwork, workspace());

        // SMV outputs need to be converted into float32 before validations.
        verifyOutputs<float>(
                convertFp16ToFp32Tensor(output, workspace()), refOutput);
    }
}
