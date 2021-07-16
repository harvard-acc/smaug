import numpy as np
import warnings

from smaug.core import types_pb2
from smaug.core import node_pb2
from smaug.python import global_vars, tensor_utils
from smaug.python.ops import common

def reorder(input_tensor, target_layout, name="reorder"):
  """Reorder the data of a given `Tensor` with the target layout.

  Args:
    input_tensor: A `Tensor`.
    target_layout: The target layout.
    name: Operator name (optional).

  Returns:
    A new `Tensor` with the layout as `target_layout`.
  """
  src_layout = input_tensor.shape.layout
  src_dims = input_tensor.shape.dims
  if src_layout == types_pb2.NCHW:
    assert (target_layout == types_pb2.NHWC or target_layout == types_pb2.NC)
    if target_layout == types_pb2.NC:
      output_tensor_dims = [src_dims[0], np.prod(src_dims[1:])]
    else:
      output_tensor_dims = [src_dims[0], src_dims[2], src_dims[3], src_dims[1]]
  elif src_layout == types_pb2.NHWC:
    assert (target_layout == types_pb2.NCHW or target_layout == types_pb2.NC)
    if target_layout == types_pb2.NC:
      output_tensor_dims = [src_dims[0], np.prod(src_dims[1:])]
    else:
      output_tensor_dims = [src_dims[0], src_dims[3], src_dims[1], src_dims[2]]
  elif (src_layout == types_pb2.NTC and target_layout == types_pb2.NCT) or (
      src_layout == types_pb2.NCT and target_layout == types_pb2.NTC):
    output_tensor_dims = [src_dims[0], src_dims[2], src_dims[1]]
  elif (src_layout == types_pb2.NC and target_layout == types_pb2.CN) or (
      src_layout == types_pb2.CN and target_layout == types_pb2.NC):
    # 2D tensor transposition.
    output_tensor_dims = [src_dims[1], src_dims[0]]
  else:
    raise ValueError(
        "Unsupported reordering %s->%s!" %
        (DataLayout.Name(src_layout), DataLayout.Name(target_layout)))

  return common.add_node(
      name=name, op=types_pb2.Reorder, input_tensors=[input_tensor],
      output_tensors_dims=[output_tensor_dims],
      output_tensor_layout=target_layout)[0]

def flatten(input_tensor, name="flatten"):
  """Flatten the data of a given `Tensor`.

  Args:
    input_tensor: A 4D `Tensor`.
    name: Operator name (optional).

  Returns:
    A 2D `Tensor` shaped as `NC`.
  """
  assert (len(input_tensor.shape.dims) == 4)
  return reorder(
      name=name, input_tensor=input_tensor, target_layout=types_pb2.NC)

def concat(input_tensors, axis=0, name="concat"):
  """Concatenate tensors into one.

  Args:
    input_tensors: Input tensor to be concatenated.
    axis: The dimension along which to concatenate.
    name: Name of the operator.

  Returns:
    A concatenated tensor.
  """
  dims = np.delete(input_tensors[0].shape.dims, axis)
  if not all([np.array_equal(np.delete(x.shape.dims, axis), dims)
              for x in input_tensors]):
    raise ValueError(
        "Tensors must have the same shape, except in axis %d along which to "
        "concatenate." % axis)
  output_tensor_dims = list(input_tensors[0].shape.dims)
  output_tensor_dims[axis] = sum(x.shape.dims[axis] for x in input_tensors)
  params = node_pb2.Params()
  params.concat_params.concat_axis = axis
  return common.add_node(
      name=name, op=types_pb2.Concat, input_tensors=input_tensors,
      output_tensors_dims=[output_tensor_dims],
      output_tensor_layout=input_tensors[0].shape.layout, params=params)[0]

def split(input_tensor, num_or_size_splits, axis=0, name="split"):
  """Split a tensor into sub tensors.

  Args:
    input_tensor: Input tensor.
    num_or_size_splits: Either an integer indicating the number of splits along
      axis or a 1D list containing the sizes of each output tensor along axis.
      If an integer, then it must evenly divide input_tensor.shape.dims[axis];
      otherwise the sum of sizes along the split axis must match that of the
      value.
    axis: The dimension to split.
    name: Name of the operator.

  Returns:
    A list of sub tensors.
  """
  splits = num_or_size_splits
  dim = input_tensor.shape.dims[axis]
  if not isinstance(num_or_size_splits, list):
    if dim % num_or_size_splits != 0:
      raise ValueError(
          "The size (%d) of the axis along which to split must divide the "
          "splits (%d)!" % (dim, num_or_size_splits))
    splits = [dim // num_or_size_splits] * num_or_size_splits
  if sum(splits) != input_tensor.shape.dims[axis]:
    raise ValueError(
        "the sum (%d) of sizes along the split axis must match that of the "
        "input (%d)!" % (sum(splits), input_tensor.shape.dims[axis]))
  if splits == [1]:
    warnings.warn(
        "Number of splits is 1 for the split operator, thus this operator is "
        "optimized out.")
    return [input_tensor]
  output_tensors_dims = []
  for s in splits:
    dims = list(input_tensor.shape.dims)
    dims[axis] = s
    output_tensors_dims.append(dims)
  params = node_pb2.Params()
  params.split_params.split_axis = axis
  return common.add_node(
      name=name, op=types_pb2.Split, input_tensors=[input_tensor],
      output_tensors_dims=output_tensors_dims,
      output_tensor_layout=input_tensor.shape.layout, params=params)

def reshape(input_tensor, shape, layout, name="reshape"):
  """ Reshape the given tensor in the same order.

  Args:
    input_tensor: Input tensor.
    shape: New shape.
    layout: New layout.
    name: Name of the operator.

  Returns:
    Tensor with the new shape.
  """
  assert np.prod(input_tensor.shape.dims) == np.prod(shape)
  return common.add_node(
      name=name, op=types_pb2.Reshape, input_tensors=[input_tensor],
      output_tensors_dims=[shape], output_tensor_layout=layout)[0]

def expand_dims(input_tensor, axis=0, name="expand_dims"):
  """Expand a tensor with an additional dimension.

  Args:
    input_tensor: Input tensor.
    axis: Dimension to expand the shape of input tensor.
    name: Name used for naming the operator.

  Returns:
    A tensor with an additional dimension inserted at index axis.
  """
  if not (input_tensor.shape.layout == types_pb2.NC and
          (axis == 1 or axis == 2)):
    raise ValueError("Currently we only support expanding NC layout.")
  output_tensor_dims = np.insert(input_tensor.shape.dims, axis, 1)
  output_tensor_layout = types_pb2.NCT if axis == 2 else types_pb2.NTC
  return reshape(input_tensor, output_tensor_dims, output_tensor_layout, name)

def squeeze(input_tensor, axis, name="squeeze"):
  """Eliminate a dimension of size 1 from a tensor.

  Args:
    input_tensor: Input tensor.
    axis: Dimension to be removed from the input tensor.
    name: Named used for naming the operator.

  Returns:
    A tensor with a dimension removed at index axis.
  """
  if input_tensor.shape.layout not in [types_pb2.NTC, types_pb2.NCT]:
    raise ValueError("Currently we only support squeezing NCT and NTC to NC.")
  output_tensor_dims = np.delete(input_tensor.shape.dims, axis)
  output_tensor_layout = types_pb2.NC
  return reshape(input_tensor, output_tensor_dims, output_tensor_layout, name)

def repeat(input_tensor, multiples, name="repeat"):
  """Construct a tensor by repeating a given tensor.

  Args:
    input_tensor: Input tensor.
    multiples: A list that represents the number of multiples in each dimension
      of the input tensor.
    name: Name of the operator.

  Returns:
    A repeated version of the input tensor.
  """
  if len(input_tensor.shape.dims) != len(multiples):
    raise ValueError(
        "The multiples of the repeat operator must have the same number of "
        "dims as the input tensor.")
  output_tensor_dims = np.multiply(input_tensor.shape.dims, multiples)
  return common.add_node(
      name=name, op=types_pb2.Repeat, input_tensors=[input_tensor],
      output_tensors_dims=[output_tensor_dims],
      output_tensor_layout=input_tensor.shape.layout)[0]

def stack(input_tensor, multiple, axis, name="stack"):
  """Expand and repeat the specified dimension of a tensor.

  Args:
    input_tensor: Input tensor.
    multiple: Number of repeats in the expanded dimension.
    axis: Dimension on which to batch.
    name: Name used for naming operators.

  Returns:
    A tensor with a new dimension.
  """
  output = expand_dims(input_tensor, axis, name=name + ":expand_dims")
  multiples = np.ones(len(output.shape.dims), dtype=np.int32)
  multiples[axis] = multiple
  output = repeat(output, multiples, name=name + ":repeat")
  return output

def unstack(input_tensor, axis, name="unstack"):
  """Unpack the specified dimension of a tensor.

  The size = 1 dimension gets squeezed out.

  Args:
    input_tensor: Input tensor.
    axis: Dimension on which to unpack.
    name: Name used for naming operators.

  Returns:
    A list of tensors with the given dimension unpacked.
  """
  split_tensors = split(
      input_tensor, input_tensor.shape.dims[axis], axis, name=name + ":split")
  outputs = []
  for i,tensor in enumerate(split_tensors):
    outputs.append(squeeze(tensor, axis, name=name + ":squeeze"))
  return outputs

def broadcast_inputs(tensor_a, tensor_b, name="broadcast_inputs"):
  """Broadcast inputs to have a compatible shape.

  This uses NumPy's broadcasting rules to make inputs of different shapes have a
  compatible shape during arithmetic operations. On each axis, the smaller
  dimension (of size 1) is broadcast across the larger dimension so that they
  have compatible shapes. Broadcasting provides a means of vectorizing
  operations.

  Args:
    tensor_a: The first input tensor.
    tensor_b: The second input tensor.
    name: Name prefix for the operators used in this function.

  Returns:
    Two new tensors with the same shape.

  Examples:

  .. code:: python

     a = np.random.rand(2, 8).astype(np.float16)
     b = np.random.rand(2, 1).astype(np.float16)
     tensor_a = Tensor(data_layout=NC, tensor_data=a)
     tensor_b = Tensor(data_layout=NC, tensor_data=b)
     # The elementwise add operator calls _broadcast_inputs() so that tensor_b
     # is broadcast in axis 1, making both inputs shaped [2, 8].
     output = add(tensor_a, tensor_b)

  .. code:: python

     a = np.random.rand(2, 16, 1, 8).astype(np.float16)
     b = np.random.rand(2, 1, 8, 8).astype(np.float16)
     tensor_a = Tensor(data_layout=NHWC, tensor_data=a)
     tensor_b = Tensor(data_layout=NHWC, tensor_data=b)
     # The elementwise mul operator calls _broadcast_inputs() so that both
     # inputs will be shaped [2, 16, 8, 8].
     output = mul(tensor_a, tensor_b)
  """
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

def padding(input_tensor, padding_size, name="padding"):
  """Construct a tensor by padding a given tensor.

  Args:
    input_tensor: Input tensor.
    padding_size: A list in the format of {dim0_begin, dim0_end, dim1_begin, 
                  dim1_end, ...} that represent number of values padded to 
                  each dimension. Note that the order of dimensions of this 
                  must align with the data layout of input_tensor.
    name: Name of the operator.

  Returns:
    A padded version of the input tensor.
  """
  src_layout = input_tensor.shape.layout
  src_dims = input_tensor.shape.dims
  if len(padding_size) != 2 * len(src_dims):
    raise ValueError("len(padding_size) should be 2x input_tensor.shape.dims")
  output_tensor_dims = [0] * len(src_dims)
  for i in range(len(src_dims)):
    output_tensor_dims[i] = src_dims[i] + padding_size[2 * i] + padding_size[2 * i+1]
  params = node_pb2.Params()
  params.padding_params.padding_size.extend(padding_size)
  return common.add_node(
      name=name, op=types_pb2.Padding, input_tensors=[input_tensor],
      output_tensors_dims=[output_tensor_dims],
      output_tensor_layout=input_tensor.shape.layout,
      params=params)[0]
