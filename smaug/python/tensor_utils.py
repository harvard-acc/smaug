from smaug.core import types_pb2
from smaug.python.tensor import Tensor

def get_tensor_data(tensor_data_array, tensor_name):
  """Find the tensor data for this tensor by its name."""
  for i in range(len(tensor_data_array.data_array)):
    if tensor_data_array.data_array[i].name == tensor_name:
      return tensor_data_array.data_array[i]
  return None

def get_padded_shape(shape):
  """Return a `TensorShapeProto` with dims padded.

  The padding is based on the alignment requirement stored in `shape`.
  """
  alignment = shape.alignment
  remainder = shape.dims[-1] % shape.alignment
  if alignment == 0 or remainder == 0:
    return shape
  shape.dims[-1] += alignment - remainder;
  return shape

def get_tensor_data_op(tensor):
  """Return the output of a data op if this tensor already has one created."""
  for node in tensor.targets:
    if node.op == types_pb2.Data:
      data_op_output = node.outputs[0]
      return data_op_output
  return None

def get_tensor_reorder_op(tensor, layout):
  """Return the output of a reorder op if this tensor already has one.

  Args:
    tensor: A Tensor.
    layout: The target layout.

  Returns:
    If found, the output tensor of the found reorder op. Otherwise, None is
    returned.
  """
  for node in tensor.targets:
    if node.op == types_pb2.Reorder and node.outputs[0].shape.layout == layout:
      reorder_op_output = node.outputs[0]
      return reorder_op_output
  return None
