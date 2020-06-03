#include <list>
#include <set>
#include <vector>
#include <boost/format.hpp>

#include "smaug/core/datatypes.h"
#include "smaug/core/typedefs.h"
#include "smaug/core/network.h"
#include "smaug/utility/utils.h"

using namespace smaug;

void Network::addOperator(Operator* op) {
    Vertex v = add_vertex(VertexProperty(op), graph);
    op->setVertex(v);
    operators[op->getName()] = op;
}

void Network::addEdge(Operator* src, Operator* dest, TensorIndices indices) {
    assert(src != dest && "Adding an edge to the node itself!");
    add_edge(src->getVertex(), dest->getVertex(), EdgeProperty(indices), graph);
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
    std::cout << hline << "\n";
    std::cout << boost::format(kLayerFormat)
                 % "Layer (type)"
                 % "Output shape"
                 % "Parameters";
    std::cout << hline << "\n";
    for (auto vertex : vertices) {
        Operator* op = get(boost::vertex_op, graph, vertex);
        op->printSummary(std::cout);
        std::cout << hline << "\n";
    }
}
