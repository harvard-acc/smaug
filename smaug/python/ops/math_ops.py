from smaug.core import types_pb2
from smaug.python.ops import common

def _math_op_common(tensor_a, tensor_b, op, name, output_tensor_dtype=None):
  if tensor_a.shape.dims != tensor_b.shape.dims:
    tensor_a, tensor_b = common.broadcast_inputs(tensor_a, tensor_b, name)
  if output_tensor_dtype == None:
    output_tensor_dtype = tensor_a.data_type
  return common.add_node(
      name=name, op=op, input_tensors=[tensor_a, tensor_b],
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
  return _math_op_common(tensor_a, tensor_b, types_pb2.EltwiseAdd, name)

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
  return _math_op_common(tensor_a, tensor_b, types_pb2.EltwiseMul, name)

def less(tensor_a, tensor_b, name="less"):
  """Returns the truth value of (tensor_a < tensor_b) element-wise.

  If the inputs have different shapes, broadcasting is used to to make the
  shapes compatible.
  """
  return _math_op_common(
      tensor_a, tensor_b, types_pb2.Less, name,
      output_tensor_dtype=types_pb2.Bool)

def less_equal(tensor_a, tensor_b, name="less_equal"):
  """Returns the truth value of (tensor_a <= tensor_b) element-wise.

  If the inputs have different shapes, broadcasting is used to to make the
  shapes compatible.
  """
  return _math_op_common(
      tensor_a, tensor_b, types_pb2.LessEqual, name,
      output_tensor_dtype=types_pb2.Bool)

def greater(tensor_a, tensor_b, name="great"):
  """Returns the truth value of (tensor_a > tensor_b) element-wise.

  If the inputs have different shapes, broadcasting is used to to make the
  shapes compatible.
  """
  return _math_op_common(
      tensor_a, tensor_b, types_pb2.Greater, name,
      output_tensor_dtype=types_pb2.Bool)

def greater_equal(tensor_a, tensor_b, name="great_equal"):
  """Returns the truth value of (tensor_a >= tensor_b) element-wise.

  If the inputs have different shapes, broadcasting is used to to make the
  shapes compatible.
  """
  return _math_op_common(
      tensor_a, tensor_b, types_pb2.GreaterEqual, name,
      output_tensor_dtype=types_pb2.Bool)
