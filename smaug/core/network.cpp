#include <list>
#include <set>
#include <vector>

#include "smaug/core/datatypes.h"
#include "smaug/core/typedefs.h"
#include "smaug/core/network.h"
#include "smaug/utility/utils.h"

using namespace smaug;

void Network::addOperator(
        Operator* op, const std::vector<Operator::IndexedOutput>& parentOps) {
    Vertex v = add_vertex(VertexProperty(op), graph);
    op->setVertex(v);
    for (int i = 0; i < parentOps.size(); i++) {
        Operator* inputOp = parentOps[i].op;
        op->setInput(inputOp->getOutput(parentOps[i].idx), i);
        Vertex sourceVertex = inputOp->getVertex();
        add_edge(sourceVertex, v, graph);
    }
    op->createAllTensors();
    operators[op->getName()] = op;
    lastOperator = op;
}

void Network::insertOperatorBetween(Operator* newOp,
                                    Operator* sourceOp,
                                    const std::vector<Operator*>& targetOps) {
    // If graph contains sourceOp->[set of targetOps], then change this to
    // sourceOp->newOp->[targetOps]. Edges that do not end in any of targetOps
    // should not be affected.
    Vertex newVertex = newOp->getVertex();
    Vertex sourceVertex = sourceOp->getVertex();
    for (Operator* targetOp : targetOps) {
        Vertex targetVertex = targetOp->getVertex();
        auto edgeToRemove = boost::edge(sourceVertex, targetVertex, graph);
        if (edgeToRemove.second)
            remove_edge(edgeToRemove.first, graph);
        add_edge(newVertex, targetVertex, EdgeProperty(NULL), graph);

        // The new op must have the same number of outputs as the source
        // operation in the same logical order. So find the original matching
        // pairs of src outputs to dest inputs and update the target ops inputs
        // with the corresponding outputs of the new op.
        auto& sourceOpOutputs = sourceOp->getOutputs();
        auto& targetOpInputs = targetOp->getInputs();
        for (int i = 0; i < sourceOpOutputs.size(); ++i) {
            auto srcTensor = sourceOpOutputs[i];
            for (int j = 0; j < targetOpInputs.size(); ++j) {
                auto dstTensor = targetOpInputs[j];
                if (srcTensor == dstTensor)
                    targetOp->setInput(newOp->getOutput(i), j);
            }
        }
    }
}

void Network::dumpDataflowGraph() const {
    std::ofstream out(name + "_dataflow_graph.dot", std::ofstream::out);
    write_graphviz(out, graph, DataflowGraphWriter(graph));
}

bool Network::validate() const {
    bool success = true;
    for (auto& iter : operators) {
        Operator* op = iter.second;
        if (!op->validate()) {
            std::cerr << "[ERROR]: " << op->getName()
                      << " was not configured correctly!\n";
            success = false;
        }
    }
    return success;
}

void Network::printSummary() const {
    static const std::string hline(
            "______________________________________________"
            "______________________________________________");
    std::list<Vertex> vertices;
    boost::topological_sort(graph, std::front_inserter(vertices));
    std::cout << hline <<"\n";
    std::cout << "Layer (type)\t\t\tOutput shape\t\tWeights shape\t\t"
                 "Parameters\n";
    std::cout << hline <<"\n";
    for (auto vertex : vertices) {
        Operator* op = get(boost::vertex_op, graph, vertex);
        op->printSummary(std::cout);
        std::cout << hline << "\n";
    }
}
