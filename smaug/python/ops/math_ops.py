from smaug.core.types_pb2 import *
from smaug.python.ops.common import *

def add(tensor_a, tensor_b, name="add"):
  """Elementwise addition.

  If the inputs have different shapes, broadcasting is used to to make the
  shapes compatible.

  Args:
    tensor_a: First input tensor.
    tensor_b: Second input tensor.
    name: Name of the operator.

  Returns:
    A tensor with the same shape as the inputs (or broadcast inputs).
  """
  if tensor_a.shape.dims != tensor_b.shape.dims:
    tensor_a, tensor_b = broadcast_inputs(tensor_a, tensor_b, name)
  return add_node(
      name=name, op=EltwiseAdd, input_tensors=[tensor_a, tensor_b],
      output_tensors_dims=[tensor_a.shape.dims],
      output_tensor_layout=tensor_a.shape.layout)[0]

def mul(tensor_a, tensor_b, name="mul"):
  """Elementwise multiplication.

  If the inputs have different shapes, broadcasting is used to to make the
  shapes compatible.

  Args:
    tensor_a: First input tensor.
    tensor_b: Second input tensor.
    name: Name of the operator.

  Returns:
    A tensor with the same shape as the inputs (or broadcast inputs).
  """
  if tensor_a.shape.dims != tensor_b.shape.dims:
    tensor_a, tensor_b = broadcast_inputs(tensor_a, tensor_b, name)
  return add_node(
      name=name, op=EltwiseMul, input_tensors=[tensor_a, tensor_b],
      output_tensors_dims=[tensor_a.shape.dims],
      output_tensor_layout=tensor_a.shape.layout)[0]

