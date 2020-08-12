from smaug.core import types_pb2
from smaug.python.ops import common

def input_data(input_tensor, name="data"):
  """Create an data node for the given tensor.

  This effectively turns a `Tensor` into a `Node`.
  """
  return common.add_node(
      name=name, op=types_pb2.Data, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout)[0]
