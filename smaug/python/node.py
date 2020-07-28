import sys
import numpy as np

from smaug.core import node_pb2
from smaug.core import types_pb2
from smaug.python import global_vars
from smaug.python import datatypes

class Node:
  def __init__(self, name, op, params=None, inputs=None, outputs=None):
    """Create a node.

    A `Node` instance contains information about its corresponding operation,
    including the operator type, parameters and input/output tensors. A `Graph`
    is made up of `Node`s. When serialized, a `NodeProto` is created.

    Args:
      name: Name of the node.
      op: `OpType` representing the operation type of the node.
      params: `Params` used by the operator (optional).
      inputs: A list of `Tensor` (optional).
      outputs: A list of `Tensor` (optional).

    Returns:
      A `Node` instance.
    """
    self._name = name
    self._op = op
    self._params = params
    self._inputs = [] if inputs is None else inputs
    self._outputs = [] if outputs is None else outputs

  @property
  def name(self):
    return self._name

  @property
  def op(self):
    return self._op

  @property
  def inputs(self):
    return self._inputs

  @property
  def outputs(self):
    return self._outputs

  def add_input(self, tensor):
    """Add an input tensor to the node.

    Args:
      tensor: A `Tensor`.
    """
    self._inputs.append(tensor)

  def add_output(self, tensor):
    """Add an output tensor to the node.

    Args:
      tensor: A `Tensor`.
    """
    self._outputs.append(tensor)

  def update_input(self, tensor, index):
    """Update the `index`th input with `tensor`.

    Args:
      tensor: A `Tensor` representing the new input.
      index: The input index.
    """
    self._inputs[index] = tensor

  def get_parents(self):
    """Get the parents of the node.

    Returns:
      A list of strings representing names of the parent nodes.
    """
    parents = []
    for tensor in self._inputs:
      if tensor.source is not None:
        parents.append(tensor.source.name)
    return parents

  def get_children(self):
    """Get the children of the node.

    Returns:
      A list of strings representing names of the children nodes.
    """
    children = []
    for tensor in self._outputs:
      for target in tensor.targets:
        children.append(target.name)
    return children

  def to_proto(self, tensor_data_array):
    """Serialize `Node` into `NodeProto`.

    Args:
      tensor_data_array: `TensorDataArray` that tensor data gets serialized
        into.

    Returns:
      A `NodeProto`.
    """
    node_proto = node_pb2.NodeProto()
    node_proto.name = self._name
    node_proto.op = self._op
    if self._params is not None:
      node_proto.params.CopyFrom(self._params)
    for tensor in self._inputs:
      if tensor.source is not None:
        node_proto.parents.append(tensor.source.name)
        node_proto.src_tensors_indices.append(tensor.source_index)
      tensor_proto = node_proto.input_tensors.add()
      tensor.to_tensor_proto(tensor_proto, tensor_data_array)
    for tensor in self._outputs:
      tensor_proto = node_proto.output_tensors.add()
      tensor.to_tensor_proto(tensor_proto, tensor_data_array)
    return node_proto
