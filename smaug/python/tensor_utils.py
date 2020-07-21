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

def from_tensor_proto(tensor_proto, tensor_data_array=None):
  """Restore a Tensor from a TensorProto.

  Args:
    tensor_proto: A TensorProto.
    tensor_data_array: a TensorDataArray that stores tensor data.

  Returns:
    A Tensor deserialized from `tensor_proto`.
  """
  name = tensor_proto.name
  data_type = tensor_proto.data_type
  tensor_data = None
  if tensor_data_array is not None:
    tensor_data_proto = get_tensor_data(tensor_data_array, name)
    if tensor_data_proto is not None:
      padded_shape = get_padded_shape(shape)
      if data_type == types_pb2.Float16:
        tensor_data = tensor_data_proto.half_data
        if padded_shape.size % 2 != 0:
          del tensor_data[-1]
      elif data_type == types_pb2.Float32:
        tensor_data = tensor_data_proto.float_data
      elif data_type == types_pb2.Float64:
        tensor_data = tensor_data_proto.double_data
      elif data_type == types_pb2.Int32:
        tensor_data = tensor_data_proto.int_data
      elif data_type == types_pb2.Int64:
        tensor_data = tensor_data_proto.int64_data
      elif data_type == types_pb2.Bool:
        tensor_data = tensor_data_proto.bool_data
      # The data retrieved from the proto is one-dimensional, so make it back to
      # shaped data.
      tensor_data.reshape(padded_shape.dims)

  tensor = Tensor(
      dims=tensor_proto.shape.dims, name=name,
      data_layout=tensor_proto.shape.layout, data_type=data_type,
      data_format=tensor_proto.data_format, tensor_data=tensor_data)
  return tensor

def get_tensor_data_op(tensor):
  """Return the output of a data op if this tensor already has one created."""
  for node in tensor.targets:
    if node.op == types_pb2.Data:
      data_op_output = from_tensor_proto(node.output_tensors[0])
      data_op_output.source = (node, 0)
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
    if node.op == types_pb2.Reorder and node.output_tensors[
        0].shape.layout == layout:
      reorder_op_output = from_tensor_proto(node.output_tensors[0])
      reorder_op_output.source = (node, 0)
      return reorder_op_output
  return None
