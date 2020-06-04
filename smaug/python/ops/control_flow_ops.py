from smaug.core.types_pb2 import *
from smaug.python.global_vars import *
from smaug.python.ops.common import *

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
  return add_node(
      name=name,
      op=Switch,
      input_tensors=[input_tensor, pred],
      output_tensors_dims=[input_tensor.shape.dims] * 2,
      output_tensor_layout=input_tensor.shape.layout)

def merge(input_tensors, name="merge"):
  """Forward the value of an available tensor from inputs to output.

  Args:
    input_tensors: Input tensors. All are dead tensor except one.

  Returns:
    A tensor that the available input tensor forwards to.
  """
  return add_node(
      name=name,
      op=Merge,
      input_tensors=input_tensors,
      output_tensors_dims=[input_tensors[0].shape.dims],
      output_tensor_layout=input_tensors[0].shape.layout)[0]
