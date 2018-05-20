#ifndef _CORE_NETWORK_H_
#define _CORE_NETWORK_H_

#include <exception>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "core/graph.h"
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

    template <typename Backend>
    void addDataLayoutTransformations(Workspace* workspace) {
        // For every operation in the graph, examine whether its output formats
        // are compatible with every one of its dependencies. If all of children
        // are incompatible, insert a reorder.
        std::vector<OperatorInsertion> reorders;
        BGL_FORALL_VERTICES(sourceVertex, graph, Graph) {
            Operator* sourceOp = get(boost::vertex_op, graph, sourceVertex);
            DataLayoutSet inputLayouts = sourceOp->getOutputDataLayouts();
            std::set<OpLayoutPair> expectedLayouts;
            BGL_FORALL_OUTEDGES(sourceVertex, edge, graph, Graph) {
                Vertex targetVertex = target(edge, graph);
                Operator* targetOp = get(boost::vertex_op, graph, targetVertex);
                DataLayoutSet outputLayouts = targetOp->getInputDataLayouts();
                if (!inputLayouts.overlapsWith(outputLayouts)) {
                    expectedLayouts.insert(
                            std::make_pair(targetOp, outputLayouts));
                }
            }
            if (expectedLayouts.empty())
                continue;
            LayoutOpsMap transformGroups =
                    findMinimalTransformGroups(expectedLayouts);
            // By construction, these are the required transformations, so it
            // doesn't matter which input data layout we pick.
            auto temp = inputLayouts.toArray();
            DataLayout sourceLayout = temp[0];
            for (auto elem : transformGroups) {
                if (elem.second.empty())
                    continue;
                DataLayout targetLayout = elem.first;
                ReorderOp<Backend>* reorder = new ReorderOp<Backend>(
                        sourceOp->getName() + "_to_" +
                                dataLayoutToStr(targetLayout),
                        targetLayout,
                        workspace);
                addOperator(reorder);
                reorders.push_back({ reorder, sourceOp, elem.second });
            }
        }
        for (auto reorderTask : reorders) {
            insertOperatorBetween(reorderTask.newOp,
                                  reorderTask.sourceOp,
                                  reorderTask.targetOps);
        }
    }

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

    LayoutOpsMap findMinimalTransformGroups(
            const std::set<OpLayoutPair>& expectedLayouts);
    void findMinimalContainingSet(LayoutOpsMap currentSet,
                                  std::set<DataLayout> remainingLayouts,
                                  std::set<OpLayoutPair> remainingOpLayoutPairs,
                                  LayoutOpsMap& minSet);


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
