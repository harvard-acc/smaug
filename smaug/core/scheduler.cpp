#include <iostream>
#include <string>
#include <vector>

#include "smaug/utility/debug_stream.h"
#include "smaug/utility/thread_pool.h"
#include "smaug/core/tensor.h"
#include "smaug/core/types.pb.h"
#include "smaug/core/scheduler.h"

namespace smaug {

Tensor* Scheduler::runNetwork() {
    std::cout << "======================================================\n";
    std::cout << "      Tiling operators of the network...\n";
    std::cout << "======================================================\n";
    for (auto nameOp : network->getOperators()) {
        Operator* op = nameOp.second;
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
    // Initialize number of pending inputs for every operator and put Data
    // operators into the ready queue.
    for (auto nameOp : network->getOperators()) {
        Operator* op = nameOp.second;
        Vertex vertex = op->getVertex();
        int numPendingInputs = boost::in_degree(vertex, network->getGraph());
        op->setNumPendingInputs(numPendingInputs);
        if (numPendingInputs == 0)
            readyQueue.push_back(op);
    }
    Tensor* output;
    {
        auto stats =
                gem5::ScopedStats(stats::kNetworkStart, stats::kNetworkEnd);
        output = scheduleReady();
    }
    return output;
}

Tensor* Scheduler::scheduleReady() {
    Tensor* output;
    for (auto op : readyQueue) {
        dout(0) << "Scheduling " << op->getName() << " ("
                << OpType_Name(op->getOpType()) << ").\n";
        maybeRunOperator(op);
        updateChildren(op);
        output = op->getOutput(0);
        dout(2) << *output << "\n";
    }
    return output;
}

void Scheduler::maybeRunOperator(Operator* op) {
    if (!op->isDead()) {
        op->run();
    } else {
        for (auto output : op->getOutputs())
            output->setDead();
    }
}

void Scheduler::updateChildren(Operator* op) {
    const Graph& graph = network->getGraph();
    Vertex vertex = op->getVertex();
    out_edge_iter outEdgeIt, outEdgeEnd;
    for (boost::tie(outEdgeIt, outEdgeEnd) = out_edges(vertex, graph);
         outEdgeIt != outEdgeEnd;
         ++outEdgeIt) {
        Vertex childVertex = target(*outEdgeIt, graph);
        Operator* child = get(boost::vertex_op, graph, childVertex);
        if (child->getNumPendingInputs() > 0) {
            child->decrNumPendingInputs();
            if (child->getNumPendingInputs() == 0)
                readyQueue.push_back(child);
        }
    }
}

}  // namespace smaug
