from graph_pb2 import *
from types_pb2 import *
from global_vars import *
from tensor import *

class Graph:
  def __init__(self, name="DefaultGraph", backend="Reference"):
    assert (backend in backend_alignment)
    self.graph = GraphProto()
    self.graph.name = name
    self.graph.backend = backend
    self.alignment = backend_alignment[backend]

  def __enter__(self):
    if get_graph() != None:
      assert False, "We only support one active graph!"
    set_graph(self)
    return self

  def __exit__(self, *args):
    clear_graph()

  def add_node(self,
               name,
               op,
               input_tensors,
               output_tensor_dims,
               output_tensor_layout=NCHW,
               output_tensor_dtype=Float32,
               output_tensor_dformat=Uncompressed,
               params=None):
    """Create a node and add it to graph.

    Args:
      name: Name of the node.
      op: Operator type.
      input_tensors: A list of input tensors of the node.
      output_tensor_dims: Dimensionality of the output tensor.
      output_tensor_layout: Layout of the output tensor.
      output_tensor_dtype: Data type of the output tensor.
      output_tensor_dformat: Storage format of the output tensor.
      params: The parameters of the node.

    Returns:
      The output tensor of the added node.
    """
    node = self.graph.nodes.add()
    node.name = name
    node.op = op

    # Add the parameter to the node.
    if params != None:
      node.params.CopyFrom(params)

    # Update the node's parents field and append it to the children field of the
    # parents nodes. Also add every input tensor to the node.
    for tensor in input_tensors:
      if tensor.source is not None:
        node.parents.append(tensor.source.name)
        tensor.source.children.append(node.name)
      input_tensor_proto = node.input_tensors.add()
      tensor.to_tensor_proto(input_tensor_proto)

    # Create the output tensor (with the node as its source), and add it to the
    # node.
    output_tensor = Tensor(
        dims=output_tensor_dims,
        name=name,
        data_layout=output_tensor_layout,
        data_type=output_tensor_dtype,
        data_format=output_tensor_dformat,
        source=node,
        alignment=self.alignment)
    output_tensor_proto = node.output_tensors.add()
    output_tensor.to_tensor_proto(output_tensor_proto)

    return output_tensor

  def write_graph(self, name=None):
    """Serialize the graph to a protobuf file.

    Args:
      name: Name of the output protobuf file. If not specified, use the graph's
            name instead.
    """
    if name == None:
      name = self.graph.name + ".pb"
    f = open(name, "w")
    f.write(self.graph.SerializeToString())
    f.close()

  def print_summary(self):
    """Print the summary of the graph.

    This function prints information of all the nodes in the graph, including a
    node's name, operator type, input/output operators and
    input/output tensors.
    """
    print "======================================================"
    print "      Summary of the network: %s (%s)" % (self.graph.name,
                                                     self.graph.backend)
    print "======================================================"
    for node in self.graph.nodes:
      print "Name: %s (%s)" % (node.name, OpType.Name(node.op))
      print "Parents:",
      for i in node.parents:
        print i,
      print "\nChildren:",
      for o in node.children:
        print o,
      print "\nInput tensors:"
      for t in node.input_tensors:
        print " ", t.name, DataType.Name(
            t.data_type), t.shape.dims, DataLayout.Name(
                t.shape.layout), "alignment(%d)" % t.shape.alignment
      print "Output tensors:"
      for t in node.output_tensors:
        print " ", t.name, DataType.Name(
            t.data_type), t.shape.dims, DataLayout.Name(
                t.shape.layout), "alignment(%d)" % t.shape.alignment
      print "------------------------------------------------------"
