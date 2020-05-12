#include <iostream>
#include <string>
#include <vector>

#include "smaug/core/backend.h"
#include "smaug/core/operator.h"
#include "smaug/core/network.h"
#include "smaug/core/tensor.h"
#include "smaug/core/workspace.h"
#include "smaug/utility/debug_stream.h"
#include "smaug/utility/thread_pool.h"
#include "smaug/core/types.pb.h"

namespace smaug {

Tensor* runNetwork(Network* network, Workspace* workspace) {
    const Graph& graph = network->getGraph();
    std::list<Vertex> vertices;
    boost::topological_sort(graph, std::front_inserter(vertices));
    std::cout << "======================================================\n";
    std::cout << "      Tiling operators of the network...\n";
    std::cout << "======================================================\n";
    for (auto v : vertices) {
        Operator* op = get(boost::vertex_op, graph, v);
        dout(0) << "Tiling " << op->getName() << " ("
                << OpType_Name(op->getOpType()) << ").\n";
        op->tile();
    }

    // We have finished loading the model and building the network, as well as
    // the tiling of all the operators. Now we can stop fast forwarding.
    gem5::switchCpu();

    fastForwardMode = false;

    // The fast-forwarding mode uses simpler CPUs, which will be switched to
    // OoO CPUs after it's done. Therefore, the initialization of the thread
    // pool must be after the fast-forwarding, otherwise the CPU IDs will be
    // incorrect.
    if (threadPool)
        threadPool->initThreadPool();

    std::cout << "======================================================\n";
    std::cout << "      Scheduling operators of the network...\n";
    std::cout << "======================================================\n";
    Tensor* output;
    {
        auto stats =
                gem5::ScopedStats(stats::kNetworkStart, stats::kNetworkEnd);
        for (auto v : vertices) {
            Operator* op = get(boost::vertex_op, graph, v);
            dout(0) << "Scheduling " << op->getName() << " ("
                    << OpType_Name(op->getOpType()) << ").\n";
            op->run();
            output = op->getOutput(0);
            dout(2) << *output << "\n";
        }
    }
    return output;
}

}  // namespace smaug
