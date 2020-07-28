import numpy as np

from smaug.core import types_pb2
from smaug.python import global_vars
from smaug.python import tensor_utils

def check_and_add_layout_transform(name, op, input_tensors):
  """ Check and perform layout transformation for the input tensors.

  This checks the input layout against the expected layout, and if a mismatch
  is found, an reorder operator will be added to transform the tensors into
  expected layouts.

  Args:
    name: Name of the operator.
    op: OpType of the operator.
    input_tensors: A list of input tensors

  Returns:
    A list of transformed input tensors, or the original input tensors if no
    layout transformation is required.
  """
  from smaug.python.ops.array_ops import reorder
  if not global_vars.get_graph().layout_trans_enabled:
    return input_tensors
  backend = global_vars.get_graph().backend
  for i in range(len(input_tensors)):
    expected_layoutset = global_vars.backend_layouts[backend][
        op].input_layoutsets[i]
    input_layout = input_tensors[i].shape.layout
    if not expected_layoutset.contains(input_layout):
      reorder_op_output = tensor_utils.get_tensor_reorder_op(
          input_tensors[i], expected_layoutset.layouts)
      if reorder_op_output is not None:
        input_tensors[i] = reorder_op_output
        continue
      input_tensors[i] = reorder(input_tensors[i], expected_layoutset.layouts)
  return input_tensors

def add_node(
    name, op, input_tensors, output_tensors_dims,
    output_tensor_layout=types_pb2.NCHW, output_tensor_dtype=None,
    output_tensor_dformat=types_pb2.Uncompressed, params=None):
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

def broadcast_inputs(tensor_a, tensor_b, name="broadcast_inputs"):
  """Broadcast inputs to have a compatible shape.

  This uses NumPy's broadcasting rules to make inputs of different shapes have a
  compatible shape during arithmetic operations. On each axis, the smaller
  dimension (of size 1) is broadcast across the larger dimension so that they
  have compatible shapes. Broadcasting provides a means of vectorizing
  operations.

  Args:
    tensor_a, tensor_b: Two input tensors.
    name: Name prefix for the operators used in this function.

  Returns:
    Two new tensors with the same shape.

  Examples:

  ```python
  a = np.random.rand(2, 8).astype(np.float16)
  b = np.random.rand(2, 1).astype(np.float16)
  tensor_a = Tensor(data_layout=NC, tensor_data=a)
  tensor_b = Tensor(data_layout=NC, tensor_data=b)
  # The elementwise add operator calls _broadcast_inputs() so that tensor_b is
  # broadcast in axis 1, making both inputs shaped [2, 8].
  output = add(tensor_a, tensor_b)
  ```

  ```python
  a = np.random.rand(2, 16, 1, 8).astype(np.float16)
  b = np.random.rand(2, 1, 8, 8).astype(np.float16)
  tensor_a = Tensor(data_layout=NHWC, tensor_data=a)
  tensor_b = Tensor(data_layout=NHWC, tensor_data=b)
  # The elementwise mul operator calls _broadcast_inputs() so that both inputs
  # will be shaped [2, 16, 8, 8].
  output = mul(tensor_a, tensor_b)
  ```
  """
  from smaug.python.ops.array_ops import repeat
  if len(tensor_a.shape.dims) != len(tensor_b.shape.dims):
    raise ValueError(
        "Cannot broadcast: tensor_a has % dimensions but tensor_b has %." %
        (len(tensor_a.shape.dims), len(tensor_b.shape.dims)))
  multiples_a = np.ones(len(tensor_a.shape.dims), dtype=np.int32)
  multiples_b = np.ones(len(tensor_a.shape.dims), dtype=np.int32)
  # Loop over the matching dimensions of the two inputs.
  for i, (a_dim, b_dim) in enumerate(
      zip(tensor_a.shape.dims, tensor_b.shape.dims)):
    if a_dim == b_dim:
      continue
    elif a_dim == 1:
      # tensor_a will be broadcast along this dimension.
      multiples_a[i] = b_dim
    elif b_dim == 1:
      # tensor_b will be broadcast along this dimension.
      multiples_b[i] = a_dim
    else:
      raise ValueError(
          "tensor_a shape %s and tensor_b shape %s are incompatible for "
          "broadcasting)" % (str(tensor_a.shape.dims), str(
              tensor_b.shape.dims)))
  if not np.all(multiples_a == 1):
    tensor_a = repeat(tensor_a, multiples_a, name=name + ":repeat_a")
  if not np.all(multiples_b == 1):
    tensor_b = repeat(tensor_b, multiples_b, name=name + ":repeat_b")
  return tensor_a, tensor_b
