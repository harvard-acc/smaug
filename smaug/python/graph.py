from __future__ import print_function

from collections import namedtuple
from google.protobuf import text_format

from smaug.core import graph_pb2
from smaug.core import types_pb2
from smaug.core import tensor_pb2
from smaug.python import global_vars
from smaug.python.tensor import Tensor

class Graph:
  def __init__(
      self, name="DefaultGraph", backend="Reference",
      mem_policy=types_pb2.AllDma):
    assert (backend in global_vars.backend_alignment)
    self.graph = graph_pb2.GraphProto()
    self._node_names = {}
    self.graph.name = name
    self.graph.backend = backend
    self.graph.mem_policy = mem_policy
    self.alignment = global_vars.backend_alignment[backend]
    # Layout transformation is enabled by default.
    self._layout_trans_enabled = True
    # This proto stores all the parameters in the network.
    self.tensor_data_array = tensor_pb2.TensorDataArray()
    self._parent_graph = None

  def __enter__(self):
    self._parent_graph = global_vars.get_graph()
    global_vars.set_graph(self)
    return self

  def __exit__(self, *args):
    # Merge the graph into its parent if it exists.
    if self._parent_graph is not None:
      self._parent_graph.merge(self)
    global_vars.set_graph(self._parent_graph)

  @property
  def backend(self):
    return self.graph.backend

  @property
  def mem_policy(self):
    return self.graph.mem_policy

  @property
  def layout_trans_enabled(self):
    return self._layout_trans_enabled

  def merge(self, other):
    """Merge another graph into this."""
    self.get_nodes().extend(other.get_nodes())
    self.tensor_data_array.data_array.extend(other.tensor_data_array.data_array)

  def add_node(
      self, name, op, input_tensors, output_tensors_dims,
      output_tensor_layout=types_pb2.NCHW,
      output_tensor_dtype=types_pb2.Float32,
      output_tensor_dformat=types_pb2.Uncompressed, params=None):
    """Create a node and add it to graph.

    Args:
      name: Name of the node. If the name is already used by another node, a
        "_N" suffix will be added.
      op: Operator type.
      input_tensors: A list of input tensors of the node.
      output_tensors_dims: A list of dims of the output tensors.
      output_tensor_layout: Layout of the output tensor.
      output_tensor_dtype: Data type of the output tensor.
      output_tensor_dformat: Storage format of the output tensor.
      params: The parameters of the node.

    Returns:
      The output tensor of the added node.
    """
    node = self.graph.nodes.add()
    node.name = self.create_unique_name(name)
    node.op = op

    # Add the parameters to the node.
    if params != None:
      node.params.CopyFrom(params)

    # Update the node's parents field, and add every input tensor to the node.
    for i,tensor in enumerate(input_tensors):
      if tensor.name == None:
        tensor.name = node.name + "/input%d" % i
      if tensor.source is not None:
        node.parents.append(tensor.source[0].name)
        node.src_tensors_indices.append(tensor.source[1])
      tensor.targets.append(node)
      input_tensor_proto = node.input_tensors.add()
      tensor.to_tensor_proto(input_tensor_proto, self.tensor_data_array)

    # Create the output tensor (with the node as its source), and add it to the
    # node.
    output_tensors = []
    for i,d in enumerate(output_tensors_dims):
      output_tensor = Tensor(
          dims=d, name="%s/output%d" % (node.name, i),
          data_layout=output_tensor_layout, data_type=output_tensor_dtype,
          data_format=output_tensor_dformat, source=(node, i),
          alignment=self.alignment)
      output_tensor_proto = node.output_tensors.add()
      output_tensor.to_tensor_proto(output_tensor_proto, self.tensor_data_array)
      output_tensors.append(output_tensor)

    return output_tensors

  def get_node(self, node_name, recursive=False):
    """Return a node in the graph proto by its name.

    Args:
      node_name: Node name.
      recursive: If true, recursively search the node in the parent graphs.

    Returns:
      A NodeProto if we find the node.
    """
    for i in range(len(self.graph.nodes)):
      if self.graph.nodes[i].name == node_name:
        return self.graph.nodes[i]
    if recursive and self._parent_graph is not None:
      return self._parent_graph.get_node(node_name, True)

  def get_nodes(self):
    """Return nodes in the graph proto."""
    return self.graph.nodes

  def get_root_graph(self):
    """Return the root graph."""
    root = self
    while root._parent_graph is not None:
      root = root._parent_graph
    return root

  def create_unique_name(self, name):
    """ Create a unique name for the node.

    Args:
      name: The base name used to create the unique name.
    """
    root = self.get_root_graph()
    new_name = name
    if name in root._node_names:
      while True:
        root._node_names[name] += 1
        new_name = "%s_%d" % (name, root._node_names[name])
        # Make sure the new name is not already used.
        if new_name not in root._node_names:
          break
    root._node_names[new_name] = 0
    return new_name

  def disable_layout_transform(self):
    """Disable automatic layout transformation.

    Note that if the backend kernels do not support the data layouts that are
    manually specified when automatic layout transformations are disabled,
    execution will fail.
    """
    self._layout_trans_enabled = False

  def enable_layout_transform(self):
    """Enable automatic layout transformation."""
    self._layout_trans_enabled = True

  def write_graph(self, name=None):
    """Serialize the graph to a protobuf file.

    Args:
      name: Name of the output protobuf file. If not specified, use the graph's
            name instead.
    """
    if name == None:
      topo_name = self.graph.name + "_topo.pbtxt"
      params_name = self.graph.name + "_params.pb"
    with open(topo_name, "w") as f_topo, open(params_name, "wb") as f_params:
      f_topo.write(text_format.MessageToString(self.graph))
      f_params.write(self.tensor_data_array.SerializeToString())

  def print_summary(self):
    """Print the summary of the graph.

    This function prints information of all the nodes in the graph, including a
    node's name, operator type, input/output operators and
    input/output tensors.
    """
    print("=================================================================")
    print("      Summary of the network: %s (%s)" % (self.graph.name,
                                                     self.graph.backend))
    print("=================================================================")
    print(
        "Host memory access policy: %s." %
        types_pb2.HostMemoryAccessPolicy.Name(self.graph.mem_policy))
    print("-----------------------------------------------------------------")
    for node in self.graph.nodes:
      print("Name: %s (%s)" % (node.name, types_pb2.OpType.Name(node.op)))
      print("Parents:", end = '')
      for i in node.parents:
        print(i, end = ' ')
      print("\nInput tensors:")
      for t in node.input_tensors:
        print(
            " ", t.name, types_pb2.DataType.Name(t.data_type), t.shape.dims,
            types_pb2.DataLayout.Name(t.shape.layout),
            "alignment(%d)" % t.shape.alignment)
      print("Output tensors:")
      for t in node.output_tensors:
        print(
            " ", t.name, types_pb2.DataType.Name(t.data_type), t.shape.dims,
            types_pb2.DataLayout.Name(t.shape.layout),
            "alignment(%d)" % t.shape.alignment)
      print("-----------------------------------------------------------------")
