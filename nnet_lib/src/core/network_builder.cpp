#include <iostream>
#include <fstream>

#include "core/backend.h"
#include "core/globals.h"
#include "core/tensor.h"
#include "core/network.h"
#include "core/workspace.h"
#include "core/graph.pb.h"
#include "core/node.pb.h"
#include "core/tensor.pb.h"
#include "core/types.pb.h"
#include "modelconf/read_model_conf.h"
#include "operators/batch_norm_op.h"
#include "operators/convolution_op.h"
#include "operators/data_op.h"
#include "operators/depthwise_convolution_op.h"
#include "operators/eltwise_add_op.h"
#include "operators/elu_op.h"
#include "operators/inner_product_op.h"
#include "operators/pooling_op.h"
#include "operators/relu_op.h"
#include "operators/reorder_op.h"
#include "operators/sigmoid_op.h"
#include "operators/softmax_op.h"
#include "operators/tanh_op.h"
#include "operators/smv/smv_convolution_op.h"
#include "utility/utils.h"
#include "utility/debug_stream.h"

using namespace smaug;
using namespace std;

// This creates a data operator for a tensor. For a data operator, the input
// tensor is simply forwarded to the output tensor. By creating a data operator
// for every tensor, we ensure that a tensor always has a source operator.
static DataOp<GlobalBackend>* createTensorOperator(
        Tensor<GlobalBackend>* tensor, Network* network, Workspace* workspace) {
    auto tensor_op = GlobalBackend::createDataOp(tensor->getName(), workspace);
    tensor_op->setData(tensor);
    network->addOperator(tensor_op);
    return tensor_op;
}

static void createAndAddOperator(const NodeProto& node,
                                 Network* network,
                                 Workspace* workspace) {
    const std::string& name = node.name();
    OpType type = node.op();

    dout(0) << "Adding " << name << " (" << OpType_Name(type) << ").\n";

    // Find all the input operators for this operator.
    std::vector<Operator*> inputs;
    for (int i = 0; i < node.parents_size(); i++) {
        std::string input_name = node.parents(i);
        inputs.push_back(network->getOperator(input_name));
    }

    if (type == OpType::Data) {
        auto input_tensor = workspace->addTensor(
                new Tensor<GlobalBackend>(node.input_tensors(0)));
        createTensorOperator(input_tensor, network, workspace);
    } else if (type == OpType::Convolution3d ||
               type == OpType::ConvolutionDepthwise) {
        ConvolutionOp<GlobalBackend>* op;
        if (type == OpType::Convolution3d)
            op = GlobalBackend::createConvolutionOp(name, workspace);
        else
            op = GlobalBackend::createDepthwiseConvolutionOp(name, workspace);
        assert(node.input_tensors_size() == 2);
        const TensorProto& filterTensorProto = node.input_tensors(1);
        auto filterTensor = workspace->addTensor(
                new Tensor<GlobalBackend>(filterTensorProto));
        DataOp<GlobalBackend>* filterTensorOp =
                createTensorOperator(filterTensor, network, workspace);
        inputs.push_back(filterTensorOp);
        assert(filterTensorProto.shape().dims_size() == 4);
        op->setWeightDims(filterTensorProto.shape().dims(2),
                          filterTensorProto.shape().dims(3),
                          filterTensorProto.shape().dims(0));
        const ConvParams& convParams = node.params().conv_params();
        assert(convParams.stride_size() == 2);
        op->setStride(convParams.stride(0), convParams.stride(1));
        op->setPadding(convParams.padding());
        network->addOperator(op, inputs);
    } else if (type == OpType::MaxPooling || type == OpType::AveragePooling) {
        PoolingOp<GlobalBackend>* op;
        if (type == MaxPooling)
            op = GlobalBackend::createMaxPoolingOp(name, workspace);
        else
            op = GlobalBackend::createAvgPoolingOp(name, workspace);
        const PoolParams& poolParams = node.params().pool_params();
        assert(poolParams.stride_size() == 2);
        assert(poolParams.pool_size_size() == 2);
        op->setPoolingSize(poolParams.pool_size(0), poolParams.pool_size(1));
        op->setPoolingStride(poolParams.stride(0), poolParams.stride(1));
        network->addOperator(op, inputs);
    } else if (type == OpType::InnerProduct) {
        auto op = GlobalBackend::createInnerProductOp(name, workspace);
        assert(node.input_tensors_size() == 2);
        const TensorProto& weightTensorProto = node.input_tensors(1);
        auto weightTensor = workspace->addTensor(
                new Tensor<GlobalBackend>(weightTensorProto));
        DataOp<GlobalBackend>* weightTensorOp =
                createTensorOperator(weightTensor, network, workspace);
        inputs.push_back(weightTensorOp);
        op->setNumOutputs(weightTensorProto.shape().dims(0));
        network->addOperator(op, inputs);
    } else if (type == OpType::Reorder) {
        auto op = GlobalBackend::createFlattenOp(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == OpType::BatchNorm) {
        auto op = GlobalBackend::createBatchNormOp(name, workspace);
        assert(node.input_tensors_size() ==
               BatchNormOp<GlobalBackend>::kNumInputs);
        for (int i = BatchNormOp<GlobalBackend>::Mean;
             i < BatchNormOp<GlobalBackend>::kNumInputs;
             i++) {
            auto tensor = workspace->addTensor(
                    new Tensor<GlobalBackend>(node.input_tensors(i)));
            DataOp<GlobalBackend>* tensor_op =
                    createTensorOperator(tensor, network, workspace);
            inputs.push_back(tensor_op);
        }
        network->addOperator(op, inputs);
    } else if (type == OpType::EltwiseAdd) {
        auto op = GlobalBackend::createEltwiseAddOp(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == OpType::ReLU) {
        auto op = GlobalBackend::createReluOp(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == OpType::LReLU) {
        // TODO: Add parameter to enable customization of this behavior.
        auto op = GlobalBackend::createReluOp(name, workspace);
        op->setSlope(0.1);
        network->addOperator(op, inputs);
    } else if (type == OpType::ELU) {
        auto op = GlobalBackend::createEluOp(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == OpType::SELU) {
        auto op = GlobalBackend::createSeluOp(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == OpType::Sigmoid) {
        auto op = GlobalBackend::createSigmoidOp(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == OpType::Tanh) {
        auto op = GlobalBackend::createTanhOp(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == OpType::HardTanh) {
        auto op = GlobalBackend::createHardTanhOp(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == OpType::UnknownOp) {
        assert(false && "Invalid operator type!");
    }
}

Network* smaug::buildNetwork(const std::string& modelFile,
                             Workspace* workspace) {
    GraphProto graph;
    fstream model(modelFile, ios::in | ios::binary);
    if (!model) {
        cout << modelFile << ": File not found." << endl;
        exit(1);
    } else if (!graph.ParseFromIstream(&model)) {
        cout << "Failed to parse the model file.\n";
        exit(1);
    }
    if (graph.backend() != GlobalBackend::Name) {
        std::cout << "The target backend " << graph.backend()
                  << " of the graph doesn't match the GlobalBackend "
                  << GlobalBackend::Name << "!\n";
        exit(1);
    }

    cout << "======================================================\n";
    cout << "      Loading the network model...\n";
    cout << "======================================================\n";
    Network* network = new Network(graph.name());
    for (int i = 0; i < graph.nodes_size(); i++) {
        const NodeProto& node = graph.nodes(i);
        createAndAddOperator(node, network, workspace);
    }

    // Allocate storage for all of the output tensors. We have filled data for
    // all the parameterizable input tensors (including the input tensor of the
    // input layer) from the model file, now we only need to allocate storage
    // for the output tensors.
    // TODO: the data type is currently fixed to float. Change that to the
    // output data type of the operator. We will have a getOutputDataType()
    // function in the operator.
    for (auto iter = network->begin(); iter != network->end(); ++iter) {
        Operator* op = iter->second;
        for (auto output : op->getOutputs()) {
            Tensor<GlobalBackend>* tensor =
                    dynamic_cast<Tensor<GlobalBackend>*>(output);
            tensor->allocateStorage<float>();
        }
    }

    smv::kSpadSize = 32*1024;

    cout << "======================================================\n";
    cout << "      Summary of the network.\n";
    cout << "======================================================\n";
    network->printSummary();
    return network;
}
