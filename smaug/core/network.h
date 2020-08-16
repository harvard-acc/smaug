#ifndef _CORE_NETWORK_H_
#define _CORE_NETWORK_H_

#include <exception>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <utility>

#include "smaug/core/typedefs.h"
#include "smaug/core/operator.h"
#include "smaug/core/workspace.h"
#include "smaug/operators/common.h"

namespace smaug {

/**
 * DataflowGraphWriter writes the current network as a dot-graph file to the
 * given ostream.
 */
class DataflowGraphWriter {
   public:
    DataflowGraphWriter(const Graph& _graph) : graph(_graph) {}
    void operator()(std::ostream& out, const Vertex& v) {
        Operator* op = get(boost::vertex_op, graph, v);
        out << "[label=\"" << op->getName() << "\"]";
    }

   protected:
    const Graph& graph;
};

/**
 * Network encapsulates all of the information SMAUG will use during execution:
 * the overall computation graph of the model, all the operators and tensors,
 * various housekeeping structures, and simulation information.
 */
class Network {
   protected:
    typedef std::map<std::string, Operator*> OperatorMap;

   public:
    Network(std::string _name) : name(_name) {}
    ~Network() {
        for (auto& op : operators)
            delete op.second;
    }

    void addOperator(Operator* op);
    void addEdge(Operator* src, Operator* dest, TensorIndices indices);
    const OperatorMap& getOperators() const { return operators; }
    Operator* getOperator(const std::string& name) {
        return operators.at(name);
    }
    const Graph& getGraph() const { return graph; }
    void dumpDataflowGraph() const;

    void printSummary() const;
    bool validate() const;
    OperatorMap::iterator begin() { return operators.begin(); }
    OperatorMap::iterator end() { return operators.end(); }

    void setSamplingInfo(const SamplingInfo& _sampling) {
        sampling = _sampling;
    }
    SamplingInfo& getSamplingInfo() { return sampling; }

   protected:
    struct OperatorInsertion {
        Operator* newOp;
        Operator* sourceOp;
        std::vector<Operator*> targetOps;
    };

    /** The dataflow graph. */
    Graph graph;

    /** Global map of operator names to their operator objects. */
    OperatorMap operators;

    /** The sampling information of the model. */
    SamplingInfo sampling;

    /** Name of the model. */
    std::string name;
};

}  // namespace smaug

#endif
