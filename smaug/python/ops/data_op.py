from smaug.core.types_pb2 import *
from smaug.python.ops.common import *

def input_data(input_tensor, name="data"):
  return add_node(
      name=name, op=Data, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout)[0]
