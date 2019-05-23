#ifndef _CORE_NETWORK_H_
#define _CORE_NETWORK_H_

#include <exception>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "core/typedefs.h"
#include "core/operator.h"
#include "core/workspace.h"

namespace smaug {

template <typename Backend>
class ReorderOp;

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

class Network {
   protected:
    typedef std::map<DataLayout, std::vector<Operator*>> LayoutOpsMap;
    typedef std::map<std::string, Operator*> OperatorMap;

   public:
    Network(std::string _name) : name(_name), lastOperator(NULL) {}
    ~Network() {
        for (auto& op : operators)
            delete op.second;
    }

    void addOperator(
            Operator* op,
            const std::vector<Operator*> parentOps = std::vector<Operator*>());
    void insertOperatorBetween(Operator* newOp,
                               Operator* sourceOp,
                               const std::vector<Operator*>& targetOps);
    Operator* getOperator(const std::string& name) {
        return operators.at(name);
    }
    Operator* getLayerLastOperator(const std::string& name) {
        return layerLastOps.at(name);
    }
    Operator* getLastOperator() const { return lastOperator; }
    const Graph& getGraph() const { return graph; }
    void dumpDataflowGraph() const;

    void printSummary() const;
    void addLayerLastOperator(std::string label, Operator* op) {
        assert(op && "Operator cannot be NULL!");
        layerLastOps[label] = op;
    }
    bool validate() const;
    OperatorMap::iterator begin() { return operators.begin(); }
    OperatorMap::iterator end() { return operators.end(); }

   protected:
    struct OperatorInsertion {
        Operator* newOp;
        Operator* sourceOp;
        std::vector<Operator*> targetOps;
    };

    typedef std::pair<Operator*, DataLayoutSet> OpLayoutPair;

    // The dataflow graph.
    Graph graph;

    // The last operator to be added to the network.
    Operator* lastOperator;

    // Global map of operator names to their operator objects.
    OperatorMap operators;

    // Map from layer name to the operator that produces its final output.i
    //
    // A layer could be comprised of multiple operators (e.g. inner product +
    // eltwise add + activation function), but in the model configuration file,
    // they are specified as a single named layer section, and subsequent layer
    // sections can specify the overall output of that layer as an input.
    OperatorMap layerLastOps;

    // Name of the model.
    std::string name;
};

}  // namespace smaug

#endif
