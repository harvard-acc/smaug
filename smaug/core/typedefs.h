#ifndef _CORE_GRAPH_H_
#define _CORE_GRAPH_H_

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/topological_sort.hpp>

namespace smaug {
class TensorBase;
class Operator;

/**
 * Additional metadata for edges in the graph.
 *
 * Edges are Tensors. This stores the input indices of the tensor in the source
 * and destination operators.
 */
struct TensorIndices {
    int srcIdx;
    int destIdx;
};

}  // namespace smaug

namespace boost {
  enum vertex_op_t { vertex_op };
  BOOST_INSTALL_PROPERTY(vertex, op);
}

typedef boost::property<boost::vertex_op_t, smaug::Operator*> VertexProperty;
typedef boost::property<boost::edge_name_t, smaug::TensorIndices> EdgeProperty;
typedef boost::adjacency_list<boost::vecS,
                              boost::vecS,
                              boost::bidirectionalS,
                              VertexProperty,
                              EdgeProperty> Graph;
typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
typedef boost::graph_traits<Graph>::edge_descriptor Edge;
typedef boost::graph_traits<Graph>::vertex_iterator vertex_iter;
typedef boost::graph_traits<Graph>::edge_iterator edge_iter;
typedef boost::graph_traits<Graph>::in_edge_iterator in_edge_iter;
typedef boost::graph_traits<Graph>::out_edge_iterator out_edge_iter;
typedef boost::property_map<Graph, boost::edge_name_t>::type MutableEdgeNameMap;
typedef boost::property_map<Graph, boost::vertex_op_t>::type
        MutableVertexNameMap;
typedef boost::property_map<Graph, boost::edge_name_t>::const_type EdgeNameMap;
typedef boost::property_map<Graph, boost::vertex_op_t>::const_type
        VertexNameMap;

#endif
