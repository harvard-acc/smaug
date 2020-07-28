from smaug.core import types_pb2
from smaug.python import global_vars
from smaug.python import tensor_utils
from smaug.python.graph import Graph
from smaug.python.ops import common

switch_op_output_ports = {"true": 1, "false": 0}

def switch(input_tensor, pred, name="switch"):
  """Forward the input to output port determined by the given predication.

  Args:
    input_tensor: Input tensor.
    pred: Predication tensor. The tensor should only contain a single boolean
      value.

  Returns:
    output_false, output_true: Two tensors representing the two branches of the
      switch. Input will only be forwarded to the taken branch.
  """
  return common.add_node(
      name=name, op=types_pb2.Switch, input_tensors=[input_tensor, pred],
      output_tensors_dims=[input_tensor.shape.dims] * 2,
      output_tensor_layout=input_tensor.shape.layout)

def merge(input_tensors, name="merge"):
  """Forward the value of an available tensor from inputs to output.

  Args:
    input_tensors: Input tensors. All are dead tensor except one.

  Returns:
    A tensor that the available input tensor forwards to.
  """
  return common.add_node(
      name=name, op=types_pb2.Merge, input_tensors=input_tensors,
      output_tensors_dims=[input_tensors[0].shape.dims],
      output_tensor_layout=input_tensors[0].shape.layout)[0]

def cond(predication, true_fn, false_fn, name="cond"):
  """A conditional operator.

  This operator provides the capability of doing if-else statement. Depending on
  the predication value, either the True or the False body of the operator will
  be executed.

  Args:
    predication: A predication tensor of value 0 or 1, determining which path to
      execute.
    true_fn: The callable to be performed if `predication` is 1.
    false_fn: The callable to be performed if `predication` is 0.

  Returns:
    The tensors returned by either true_fn or false_fn.
  """

  def _insert_switch_nodes(predication, branch_result, graph):
    """Insert switch nodes for external tensors in the subgraph.

    An external tensor is a tensor that comes from a node outside this graph,
    this adds switch nodes for every external tensor in `graph`.

    Args:
      predication: The predication tensor used for determining the deadness of
        switch node results.
      branch_result: String value of "true" or "false", representing which
        result of the switch nodes to use.
      graph: A `GraphProto` that represents a branch of the conditional.
    """
    if branch_result not in ["true", "false"]:
      raise ValueError(
          "Use either 'true' or 'false' to indicate the output of the switch "
          "nodes.")
    nodes = [node for node in graph.get_nodes() if node.op != types_pb2.Data]
    # This keeps track of all the tensors that come from nodes in the graph.
    internal_tensors = set()
    for node in nodes:
      internal_tensors.update(set([tensor.name for tensor in node.outputs]))
    for node in nodes:
      for i, tensor in enumerate(node.inputs):
        # If any input tensor of the graph appear in the graph workspace, then
        # this tensor is an external to the graph and we create a switch node
        # for it.
        # Don't create switch node for an existing one.
        if node.op == types_pb2.Switch:
          continue
        if tensor.name not in internal_tensors:
          switch_result = switch(
              tensor, predication)[switch_op_output_ports[branch_result]]
          # Update the node's input with the switch node result.
          node.update_input(switch_result, i)

  cur_graph = global_vars.get_graph()
  backend = cur_graph.backend
  mem_policy = cur_graph.mem_policy
  name = cur_graph.create_unique_name(name)

  # Build the subgraph for the true branch.
  with Graph(name="%s_true_branch" % name, backend=backend,
             mem_policy=mem_policy) as subgraph_t:
    res_t = true_fn()
    if not isinstance(res_t, (list, tuple)):
      res_t = [res_t]
    _insert_switch_nodes(predication, "true", subgraph_t)

  # Build the subgraph for the false branch.
  with Graph(name="%s_false_branch" % name, backend=backend,
             mem_policy=mem_policy) as subgraph_f:
    res_f = false_fn()
    if not isinstance(res_f, (list, tuple)):
      res_f = [res_f]
    _insert_switch_nodes(predication, "false", subgraph_f)

  # Add the merge nodes for the outputs.
  merges = [merge([t, f]) for (t, f) in zip(res_t, res_f)]
  return merges
