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
