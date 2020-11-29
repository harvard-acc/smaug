import numpy as np

from smaug.core import types_pb2
from smaug.python import global_vars
from smaug.python import tensor_utils

def add_node(
    name, op, input_tensors, output_tensors_dims,
    output_tensor_layout=types_pb2.NCHW, output_tensor_dtype=None,
    output_tensor_dformat=types_pb2.Uncompressed, params=None):
  """Adds a new node to the current Graph.

  Args:
    name: Name of the new operator. If another operator in the Graph already
       has this name, a unique suffix is automatically appended.
    op: OpType of the operator.
    input_tensors: List of all input tensors.
    output_tensors_dims: List of the dimensions of all the output tensors.
    output_tensor_layout: The expected data layout of the output tensors. If
       not provided, it will use the layout of the first input tensor.
    output_tensor_dtype: The data type of the output tensor elements. If not
       provided, the data type of the first input tensor will be used.
    output_tensor_dformat: The data format of the output tensor. The only
       supported option is uncompressed data. Compressed formats may be added
       at some later time.
    params: A smaug.Params protobuf containing any additional parameters for
       this operator.

  Returns:
    A list of output tensors.
  """
  if global_vars.get_graph() == None:
    assert False, "No available active graph!"
  if output_tensor_dtype == None:
    output_tensor_dtype = input_tensors[0].data_type
  if output_tensor_layout == types_pb2.X:
    output_tensor_layout = input_tensors[0].shape.layout

  # If any input tensor doesn't have a source operator, we create a DataOp
  # for it. This makes the deserializing a lot easier in the C++ core. To avoid
  # an infinite loop, don't create a new data op if the node to be added is a
  # data op.
  for i in range(len(input_tensors)):
    if input_tensors[i].source == None and op != types_pb2.Data:
      data_op_output = tensor_utils.get_tensor_data_op(input_tensors[i])
      if data_op_output is not None:
        input_tensors[i] = data_op_output
        continue
      input_tensors[i] = global_vars.get_graph().add_node(
          name="data", op=types_pb2.Data, input_tensors=[input_tensors[i]],
          output_tensors_dims=[input_tensors[i].shape.dims],
          output_tensor_layout=input_tensors[i].shape.layout,
          output_tensor_dtype=input_tensors[i].data_type,
          output_tensor_dformat=input_tensors[i].data_format)[0]
  return global_vars.get_graph().add_node(
      name=name, op=op, input_tensors=input_tensors,
      output_tensors_dims=output_tensors_dims,
      output_tensor_layout=output_tensor_layout,
      output_tensor_dtype=output_tensor_dtype,
      output_tensor_dformat=output_tensor_dformat, params=params)
