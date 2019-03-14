#include <iostream>
#include <fstream>

#include "core/backend.h"
#include "core/tensor.h"
#include "core/network.h"
#include "core/network_builder.h"
#include "core/workspace.h"
#include "core/graph.pb.h"
#include "core/node.pb.h"
#include "core/tensor.pb.h"
#include "core/types.pb.h"
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

// Create an operator by deserializing a node in the graph, and add it to the
// network.
template <typename Backend>
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
        auto inputTensor =
                workspace->addTensor(new Tensor(node.input_tensors(0)));
        auto inputTensorOp =
                Backend::createDataOp(inputTensor->getName(), workspace);
        inputTensorOp->setData(inputTensor);
        network->addOperator(inputTensorOp);
    } else if (type == OpType::Convolution3d ||
               type == OpType::ConvolutionDepthwise) {
        ConvolutionOp<Backend>* op;
        if (type == OpType::Convolution3d)
            op = Backend::createConvolutionOp(name, workspace);
        else
            op = Backend::createDepthwiseConvolutionOp(name, workspace);
        assert(node.input_tensors_size() == 2);
        const TensorProto& filterTensorProto = node.input_tensors(1);
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
        PoolingOp<Backend>* op;
        if (type == MaxPooling)
            op = Backend::createMaxPoolingOp(name, workspace);
        else
            op = Backend::createAvgPoolingOp(name, workspace);
        const PoolParams& poolParams = node.params().pool_params();
        assert(poolParams.stride_size() == 2);
        assert(poolParams.pool_size_size() == 2);
        op->setPoolingSize(poolParams.pool_size(0), poolParams.pool_size(1));
        op->setPoolingStride(poolParams.stride(0), poolParams.stride(1));
        network->addOperator(op, inputs);
    } else if (type == OpType::InnerProduct) {
        auto op = Backend::createInnerProductOp(name, workspace);
        assert(node.input_tensors_size() == 2);
        const TensorProto& weightTensorProto = node.input_tensors(1);
        op->setNumOutputs(weightTensorProto.shape().dims(0));
        network->addOperator(op, inputs);
    } else if (type == OpType::Reorder) {
        DataLayout targetLayout = node.output_tensors(0).shape().layout();
        ReorderOp<Backend>* op;
        if (targetLayout == NC) {
            op = Backend::createFlattenOp(name, workspace);
        } else {
            op = Backend::createReorderOp(name, workspace);
            op->setTargetLayout(node.output_tensors(0).shape().layout());
        }
        network->addOperator(op, inputs);
    } else if (type == OpType::BatchNorm) {
        auto op = Backend::createBatchNormOp(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == OpType::EltwiseAdd) {
        auto op = Backend::createEltwiseAddOp(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == OpType::ReLU) {
        auto op = Backend::createReluOp(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == OpType::LReLU) {
        // TODO: Add parameter to enable customization of this behavior.
        auto op = Backend::createReluOp(name, workspace);
        op->setSlope(0.1);
        network->addOperator(op, inputs);
    } else if (type == OpType::ELU) {
        auto op = Backend::createEluOp(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == OpType::SELU) {
        auto op = Backend::createSeluOp(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == OpType::Sigmoid) {
        auto op = Backend::createSigmoidOp(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == OpType::Tanh) {
        auto op = Backend::createTanhOp(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == OpType::HardTanh) {
        auto op = Backend::createHardTanhOp(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == OpType::UnknownOp) {
        assert(false && "Invalid operator type!");
    }
}

// Create the network by deserializing the graph stored in the
// protobuf model.
template <typename Backend>
static Network* createNetworkFromProto(const GraphProto& graph,
                                       Workspace* workspace) {
    Network* network = new Network(graph.name());
    for (int i = 0; i < graph.nodes_size(); i++) {
        const NodeProto& node = graph.nodes(i);
        createAndAddOperator<Backend>(node, network, workspace);
    }
    return network;
}

// Allocate storage for all of the output tensors. We have filled data for
// all the parameterizable input tensors (including the input tensor of the
// input layer) from the model file, now we only need to allocate storage
// for the output tensors.
// TODO: the data type is currently fixed to float. Change that to the
// output data type of the operator. We will have a getOutputDataType()
// function in the operator.
static void allocateTensorStorage(Network* network) {
    for (auto iter = network->begin(); iter != network->end(); ++iter) {
        Operator* op = iter->second;
        for (auto output : op->getOutputs()) {
            Tensor* tensor = dynamic_cast<Tensor*>(output);
            tensor->allocateStorage<float>();
        }
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

    cout << "======================================================\n";
    cout << "      Loading the network model...\n";
    cout << "======================================================\n";
    Network* network = nullptr;
    if (graph.backend() == ReferenceBackend::Name) {
        network = createNetworkFromProto<ReferenceBackend>(graph, workspace);
    } else if (graph.backend() == SmvBackend::Name) {
        network = createNetworkFromProto<SmvBackend>(graph, workspace);
    } else {
        assert(false && "Unknown backend!");
    }

    allocateTensorStorage(network);

    smv::kSpadSize = 32*1024;

    cout << "======================================================\n";
    cout << "      Summary of the network.\n";
    cout << "======================================================\n";
    network->printSummary();
    return network;
}
