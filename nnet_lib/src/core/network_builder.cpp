#include <iostream>
#include <fstream>
#include <fcntl.h>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "core/backend.h"
#include "core/tensor.h"
#include "core/network.h"
#include "core/network_builder.h"
#include "core/workspace.h"
#include "core/graph.pb.h"
#include "core/node.pb.h"
#include "core/tensor.pb.h"
#include "core/types.pb.h"
#include "operators/common.h"
#include "operators/batch_norm_op.h"
#include "operators/convolution_op.h"
#include "operators/data_op.h"
#include "operators/depthwise_convolution_op.h"
#include "operators/eltwise_add_op.h"
#include "operators/eltwise_mul_op.h"
#include "operators/elu_op.h"
#include "operators/inner_product_op.h"
#include "operators/pooling_op.h"
#include "operators/relu_op.h"
#include "operators/reorder_op.h"
#include "operators/concat_op.h"
#include "operators/split_op.h"
#include "operators/reshape_op.h"
#include "operators/sigmoid_op.h"
#include "operators/softmax_op.h"
#include "operators/tanh_op.h"
#include "operators/smv/smv_convolution_op.h"
#include "operators/smv/smv_inner_product_op.h"
#include "operators/smv/smv_pooling_op.h"
#include "operators/smv/smv_batch_norm_op.h"
#include "operators/smv/smv_relu_op.h"
#include "operators/smv/smv_elu_op.h"
#include "operators/smv/smv_tanh_op.h"
#include "operators/smv/smv_sigmoid_op.h"
#include "operators/smv/smv_eltwise_add_op.h"
#include "operators/smv/smv_eltwise_mul_op.h"
#include "utility/utils.h"
#include "utility/debug_stream.h"

using namespace smaug;
using namespace std;

ActivationInfo getActivationInfo(const ActivationParams& params) {
    ActivationInfo actInfo;
    OpType opType = params.activation();
    switch (opType) {
        case OpType::ReLU:
            actInfo.function = activation_type::RELU;
            break;
        case OpType::LReLU:
            actInfo.function = activation_type::LRELU;
            actInfo.params.slope = params.lrelu_params().slope();
            break;
        case OpType::ELU:
            actInfo.function = activation_type::ELU;
            actInfo.params.alpha = params.elu_params().alpha();
            break;
        case OpType::SELU:
            actInfo.function = activation_type::SELU;
            actInfo.params.alpha = params.elu_params().alpha();
            actInfo.params.lambda = params.elu_params().lambda_param();
            break;
        case OpType::Tanh:
            actInfo.function = activation_type::TANH;
            break;
        case OpType::HardTanh:
            actInfo.function = activation_type::HARD_TANH;
            actInfo.params.min = params.hard_tanh_params().min();
            actInfo.params.max = params.hard_tanh_params().max();
            break;
        case OpType::Sigmoid:
            actInfo.function = activation_type::SIGMOID;
            break;
        case OpType::Softmax:
            actInfo.function = activation_type::SOFTMAX;
        default:
            actInfo.function = activation_type::NO_ACTIVATION;
    }
    return actInfo;
}

// Create an operator by deserializing a node in the graph, and add it to the
// network.
template <typename Backend>
static void createAndAddOperator(const NodeProto& node,
                                 const TensorDataArray& tensorDataArray,
                                 HostMemoryAccessPolicy memPolicy,
                                 Network* network,
                                 Workspace* workspace) {
    const std::string& name = node.name();
    OpType type = node.op();

    dout(0) << "Adding " << name << " (" << OpType_Name(type) << ").\n";

    // Find all the input operators for this operator.
    std::vector<Operator::IndexedOutput> inputs;
    for (int i = 0; i < node.parents_size(); i++) {
        std::string input_name = node.parents(i);
        int tensor_idx = node.src_tensors_indices(i);
        inputs.push_back({ network->getOperator(input_name), tensor_idx });
    }

    if (type == OpType::Data) {
        // Find the tensor data from the tensor data array.
        TensorData tensorData;
        for (int i = 0; i < tensorDataArray.data_array_size(); i++) {
            if (tensorDataArray.data_array(i).name() ==
                node.input_tensors(0).name()) {
                tensorData = tensorDataArray.data_array(i);
                break;
            }
        }
        auto inputTensor = workspace->addTensor(
                new Tensor(node.input_tensors(0), tensorData));
        auto inputTensorOp = Backend::createDataOp(name, workspace);
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
        const TensorShapeProto& shapeProto = filterTensorProto.shape();
        assert(shapeProto.dims_size() == 4);
        if (shapeProto.layout() == NCHW) {
            op->setWeightDims(
                    shapeProto.dims(2), shapeProto.dims(3), shapeProto.dims(0));
        } else {
            op->setWeightDims(
                    shapeProto.dims(1), shapeProto.dims(2), shapeProto.dims(0));
        }
        const ConvParams& convParams = node.params().conv_params();
        assert(convParams.stride_size() == 2);
        op->setStride(convParams.stride(0), convParams.stride(1));
        op->setPadding(convParams.padding());
        op->setActivation(getActivationInfo(node.params().act_params()));
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
        if (weightTensorProto.shape().layout() == NC)
            op->setNumOutputs(weightTensorProto.shape().dims(0));
        else
            op->setNumOutputs(weightTensorProto.shape().dims(1));
        op->setActivation(getActivationInfo(node.params().act_params()));
        network->addOperator(op, inputs);
    } else if (type == OpType::Reorder) {
        DataLayout srcLayout = node.input_tensors(0).shape().layout();
        DataLayout targetLayout = node.output_tensors(0).shape().layout();
        ReorderOp<Backend>* op;
        if (node.input_tensors(0).shape().dims_size() == 4 &&
            (targetLayout == NC || targetLayout == CN)) {
            op = Backend::createFlattenOp(name, workspace);
        } else {
            op = Backend::createReorderOp(name, workspace);
            op->setTargetLayout(node.output_tensors(0).shape().layout());
        }
        network->addOperator(op, inputs);
    } else if (type == OpType::Concat) {
        auto op = Backend::createConcatOp(name, workspace);
        op->setNumInputs(node.input_tensors_size());
        op->setConcatAxis(node.params().concat_params().concat_axis());
        network->addOperator(op, inputs);
    } else if (type == OpType::Split) {
        auto op = Backend::createSplitOp(name, workspace);
        int axis = node.params().split_params().split_axis();
        std::vector<int> splits;
        for (const auto& tensor : node.output_tensors())
            splits.push_back(tensor.shape().dims(axis));
        op->setSplits(splits);
        op->setSplitAxis(axis);
        network->addOperator(op, inputs);
    } else if (type == OpType::Reshape) {
        auto op = Backend::createReshapeOp(name, workspace);
        const TensorShapeProto& shapeProto = node.output_tensors(0).shape();
        std::vector<int> shape(
                shapeProto.dims().begin(), shapeProto.dims().end());
        DataLayout layout = shapeProto.layout();
        op->setShape(shape, layout);
        network->addOperator(op, inputs);
    } else if (type == OpType::BatchNorm) {
        auto op = Backend::createBatchNormOp(name, workspace);
        op->setActivation(getActivationInfo(node.params().act_params()));
        network->addOperator(op, inputs);
    } else if (type == OpType::EltwiseAdd) {
        auto op = Backend::createEltwiseAddOp(name, workspace);
        network->addOperator(op, inputs);
    } else if (type == OpType::EltwiseMul) {
        auto op = Backend::createEltwiseMulOp(name, workspace);
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

    Operator* op = network->getOperator(name);
    // Set the sampling info for the operator if it supports sampling.
    if (op->isSamplingSupported())
        op->setSamplingInfo(network->getSamplingInfo());
    // Set the memory access types for the operator's data.
    if (memPolicy == HostMemoryAccessPolicy::AllDma) {
        op->setInputsMemType(MemoryType::dma);
        op->setWeightsMemType(MemoryType::dma);
        op->setOutputsMemType(MemoryType::dma);
    } else if (memPolicy == HostMemoryAccessPolicy::AllAcp) {
        op->setInputsMemType(MemoryType::acp);
        op->setWeightsMemType(MemoryType::acp);
        op->setOutputsMemType(MemoryType::acp);
    } else if (memPolicy == HostMemoryAccessPolicy::AllCache) {
        op->setInputsMemType(MemoryType::cache);
        op->setWeightsMemType(MemoryType::cache);
        op->setOutputsMemType(MemoryType::cache);
    } else if (memPolicy == HostMemoryAccessPolicy::AllAcpWithDmaForWeights) {
        op->setInputsMemType(MemoryType::acp);
        op->setWeightsMemType(MemoryType::dma);
        op->setOutputsMemType(MemoryType::acp);
    } else if (memPolicy == HostMemoryAccessPolicy::UnknownMemoryPolicy) {
        assert(false && "Invalid host memory access policy!");
    }

    // Allocate storage for the output tensors of the newly added operator. We
    // have filled data for all the parameterizable input tensors from the model
    // file, now we only need to allocate storage for the output tensors.
    DataType dataType = node.input_tensors(0).data_type();
    for (auto output : op->getOutputs()) {
        Tensor* tensor = dynamic_cast<Tensor*>(output);
        tensor->allocateStorage(dataType);
    }
}

// Create the network by deserializing the graph stored in the
// protobuf model.
template <typename Backend>
static Network* createNetworkFromProto(const GraphProto& graph,
                                       const TensorDataArray& tensorDataArray,
                                       SamplingInfo& sampling,
                                       Workspace* workspace) {
    Network* network = new Network(graph.name());
    network->setSamplingInfo(sampling);
    for (int i = 0; i < graph.nodes_size(); i++) {
        const NodeProto& node = graph.nodes(i);
        createAndAddOperator<Backend>(
                node, tensorDataArray, graph.mem_policy(), network, workspace);
    }
    return network;
}

Network* smaug::buildNetwork(const std::string& modelTopo,
                             const std::string& modelParams,
                             SamplingInfo& sampling,
                             Workspace* workspace) {
    // Parse the network topology from the protobuf text file.
    GraphProto graph;
    int modelTopoDescriptor = open(modelTopo.c_str(), O_RDONLY);
    if (modelTopoDescriptor < 0) {
        cout << modelTopo << ": network topology file not found." << endl;
        exit(1);
    }
    google::protobuf::io::FileInputStream modelTopoInput(modelTopoDescriptor);
    if (!google::protobuf::TextFormat::Parse(&modelTopoInput, &graph)) {
        cout << "Failed to parse the network topology file!" << endl;
        exit(1);
    }
    // Parse the network parameters from the protobuf binary file.
    TensorDataArray tensorDataArray;
    fstream modelParamsFile(modelParams, ios::in | ios::binary);
    if (!modelParamsFile) {
        cout << modelParams << ": network parameters file not found." << endl;
        exit(1);
    } else if (!tensorDataArray.ParseFromIstream(&modelParamsFile)) {
        cout << "Failed to parse the network parameters file.\n";
        exit(1);
    }

    cout << "======================================================\n";
    cout << "      Loading the network model...\n";
    cout << "======================================================\n";
    Network* network = nullptr;
    if (graph.backend() == ReferenceBackend::Name) {
        network = createNetworkFromProto<ReferenceBackend>(
                graph, tensorDataArray, sampling, workspace);
    } else if (graph.backend() == SmvBackend::Name) {
        network = createNetworkFromProto<SmvBackend>(
                graph, tensorDataArray, sampling, workspace);
    } else {
        assert(false && "Unknown backend!");
    }

    cout << "======================================================\n";
    cout << "      Summary of the network.\n";
    cout << "======================================================\n";
    network->printSummary();
    return network;
}
