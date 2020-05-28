from smaug.core.types_pb2 import *
from smaug.python.ops.common import *

def _math_op_common(tensor_a, tensor_b, op, name, output_tensor_dtype=None):
  if tensor_a.shape.dims != tensor_b.shape.dims:
    tensor_a, tensor_b = broadcast_inputs(tensor_a, tensor_b, name)
  if output_tensor_dtype == None:
    output_tensor_dtype = tensor_a.data_type
  return add_node(
      name=name,
      op=op,
      input_tensors=[tensor_a, tensor_b],
      output_tensors_dims=[tensor_a.shape.dims],
      output_tensor_layout=tensor_a.shape.layout,
      output_tensor_dtype=output_tensor_dtype)[0]

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
  return _math_op_common(tensor_a, tensor_b, EltwiseAdd, name)

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
  return _math_op_common(tensor_a, tensor_b, EltwiseMul, name)

def less(tensor_a, tensor_b, name="less"):
  """Returns the truth value of (tensor_a < tensor_b) element-wise.

  If the inputs have different shapes, broadcasting is used to to make the
  shapes compatible.
  """
  return _math_op_common(
      tensor_a, tensor_b, Less, name, output_tensor_dtype=Bool)

def less_equal(tensor_a, tensor_b, name="less_equal"):
  """Returns the truth value of (tensor_a <= tensor_b) element-wise.

  If the inputs have different shapes, broadcasting is used to to make the
  shapes compatible.
  """
  return _math_op_common(
      tensor_a, tensor_b, LessEqual, name, output_tensor_dtype=Bool)

def greater(tensor_a, tensor_b, name="great"):
  """Returns the truth value of (tensor_a > tensor_b) element-wise.

  If the inputs have different shapes, broadcasting is used to to make the
  shapes compatible.
  """
  return _math_op_common(
      tensor_a, tensor_b, Greater, name, output_tensor_dtype=Bool)

def greater_equal(tensor_a, tensor_b, name="great_equal"):
  """Returns the truth value of (tensor_a >= tensor_b) element-wise.

  If the inputs have different shapes, broadcasting is used to to make the
  shapes compatible.
  """
  return _math_op_common(
      tensor_a, tensor_b, GreaterEqual, name, output_tensor_dtype=Bool)
