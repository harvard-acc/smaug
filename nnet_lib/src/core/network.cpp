#include <list>
#include <set>
#include <vector>

#include "core/datatypes.h"
#include "core/globals.h"
#include "core/graph.h"
#include "core/network.h"
#include "utility/utils.h"

using namespace smaug;

void Network::addOperator(Operator* op,
                          const std::vector<Operator*> parentOps) {
    Vertex v = add_vertex(VertexProperty(op), graph);
    op->setVertex(v);
    for (int i = 0; i < parentOps.size(); i++) {
        Operator* inputOp = parentOps[i];
        op->setInput(inputOp->getOutput(0), i);
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
    }
}

Network::LayoutOpsMap Network::findMinimalTransformGroups(
    const std::set<OpLayoutPair>& expectedLayouts) {
    // We have a list of sets of supported data layouts. The goal is to find
    // the minimum number of unique data layouts such that every set has at
    // least one element.
    // {NCHW, NHWC}, {NCHW}, {NC}, {X} -> should produce {NCHW, NC}.
    // remainingOpLayoutPairs.remove(j);
    std::set<DataLayout> allPossibleLayouts{
        DataLayout::NCHW, DataLayout::NHWC, DataLayout::NC
    };
    LayoutOpsMap assignment;
    for (auto layout : allPossibleLayouts)
        assignment[layout] = std::vector<Operator*>();
    findMinimalContainingSet(assignment,
                             allPossibleLayouts,
                             expectedLayouts,
                             assignment);
    return assignment;
}

void Network::findMinimalContainingSet(
        LayoutOpsMap currentSet,
        std::set<DataLayout> remainingLayouts,
        std::set<OpLayoutPair> remainingOpLayoutPairs,
        LayoutOpsMap& minSet) {
    if (remainingOpLayoutPairs.empty()) {
        if (currentSet.size() <= minSet.size())
            minSet = currentSet;
        return;
    }
    for (auto layoutIt = remainingLayouts.begin();
         layoutIt != remainingLayouts.end();) {
        DataLayout currLayout = *layoutIt;
        std::set<OpLayoutPair> newRemaining;
        for (auto pairIt = remainingOpLayoutPairs.begin();
             pairIt != remainingOpLayoutPairs.end();
             ++pairIt) {
            auto p = *pairIt;
            if (p.second.contains(currLayout)) {
                currentSet[currLayout].push_back(p.first);
            } else {
                newRemaining.insert(p);
            }
        }
        layoutIt = remainingLayouts.erase(layoutIt);
        findMinimalContainingSet(currentSet,
                                 remainingLayouts,
                                 newRemaining,
                                 minSet);
        // Restore currentSet.
        currentSet[currLayout].clear();
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
