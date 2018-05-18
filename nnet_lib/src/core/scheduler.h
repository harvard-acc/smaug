#include <string>
#include <vector>

#include "core/backend.h"
#include "core/operator.h"
#include "core/network.h"
#include "core/tensor.h"
#include "core/workspace.h"

namespace smaug {

template <typename Backend>
void runNetwork(Network* network, Workspace* workspace) {
    const Graph& graph = network->getGraph();
    std::list<Vertex> vertices;
    boost::topological_sort(graph, std::front_inserter(vertices));
    for (auto v : vertices) {
        Operator* op = get(boost::vertex_op, graph, v);
        op->run();
    }
}

}  // namespace smaug
