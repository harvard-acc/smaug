import numpy as np
from types_pb2 import *
from node_pb2 import *
from global_vars import *
from tensor import *

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
  backend = get_graph().graph.backend
  expected_layoutset = backend_layouts[backend][op].input_layoutset
  for i in range(len(input_tensors)):
    input_layout = input_tensors[i].shape.layout
    if not expected_layoutset.contains(input_layout):
      input_tensors[i] = reorder("%s->%s" % (input_tensors[i].name, name),
                                 input_tensors[i], expected_layoutset.layouts)
  return input_tensors

def get_output_layout(op):
  """ Get the expected output layout for the operator.

  Args:
    op: OpType of the operator.

  Returns:
    The expected layout type of the operator.
  """
  backend = get_graph().graph.backend
  return backend_layouts[backend][op].output_layoutset.layouts

def add_node(name,
             op,
             input_tensors,
             output_tensor_dims,
             output_tensor_layout=NCHW,
             output_tensor_dtype=None,
             output_tensor_dformat=Uncompressed,
             params=None):
  if get_graph() == None:
    assert False, "No available active graph!"
  if output_tensor_dtype == None:
    output_tensor_dtype = input_tensors[0].data_type
  if output_tensor_layout == X:
    output_tensor_layout = input_tensors[0].shape.layout

  # If any input tensor doesn't have a source operator, we create a DataOp
  # for it. This makes the deserializing a lot easier in the C++ core. Note
  # that we don't need to create a DataOp for input_data.
  for i in range(len(input_tensors)):
    if input_tensors[i].source == None and op != Data:
      input_tensors[i] = get_graph().add_node(
          name=input_tensors[i].name,
          op=Data,
          input_tensors=[input_tensors[i]],
          output_tensor_dims=input_tensors[i].shape.dims,
          output_tensor_layout=input_tensors[i].shape.layout,
          output_tensor_dtype=output_tensor_dtype,
          output_tensor_dformat=output_tensor_dformat)
  return get_graph().add_node(name, op, input_tensors, output_tensor_dims,
                              output_tensor_layout, output_tensor_dtype,
                              output_tensor_dformat, params)

def input_data(name, input_tensor):
  input_tensor.name = name
  return add_node(
      name=name,
      op=Data,
      input_tensors=[input_tensor],
      output_tensor_dims=input_tensor.shape.dims,
      output_tensor_layout=input_tensor.shape.layout)

def to_padding_type(padding):
  if padding == "same":
    return SamePadding
  elif padding == "valid":
    return ValidPadding
  else:
    return UnknownPadding

def convolution(name, input_tensor, filter_tensor, stride, padding):
  def compute_output_dim(input_dim, weight_dim, stride, padding):
    pad = 0
    if to_padding_type(padding) == SamePadding:
      pad = weight_dim - 1
    return (input_dim - weight_dim + pad) / stride + 1

  filter_tensor.name = name + "/kernels"

  input_tensor, filter_tensor = check_and_add_layout_transform(
      name=name, op=Convolution3d, input_tensors=[input_tensor, filter_tensor])

  row_idx = 2 if input_tensor.shape.layout == NCHW else 1
  col_idx = 3 if input_tensor.shape.layout == NCHW else 2
  output_rows = compute_output_dim(input_tensor.shape.dims[row_idx],
                                   filter_tensor.shape.dims[2], stride[0],
                                   padding)
  output_cols = compute_output_dim(input_tensor.shape.dims[col_idx],
                                   filter_tensor.shape.dims[3], stride[1],
                                   padding)
  output_layout = get_output_layout(Convolution3d)
  if output_layout == NCHW:
    output_tensor_dims = [
        input_tensor.shape.dims[0], filter_tensor.shape.dims[0], output_rows,
        output_cols
    ]
  elif output_layout == NHWC:
    output_tensor_dims = [
        input_tensor.shape.dims[0], output_rows, output_cols,
        filter_tensor.shape.dims[0]
    ]
  else:
    assert False, "Unsupported output layout!"
  params = Params()
  params.conv_params.padding = to_padding_type(padding)
  params.conv_params.stride.extend(stride)

  return add_node(
      name=name,
      op=Convolution3d,
      input_tensors=[input_tensor, filter_tensor],
      output_tensor_dims=output_tensor_dims,
      output_tensor_layout=output_layout,
      params=params)

def relu(name, input_tensor):
  return add_node(
      name=name,
      op=ReLU,
      input_tensors=[input_tensor],
      output_tensor_dims=input_tensor.shape.dims,
      output_tensor_layout=input_tensor.shape.layout)

def batch_norm(name, input_tensor, mean_tensor, var_tensor, gamma_tensor,
               beta_tensor):
  assert (len(mean_tensor.shape.dims) == 1 and len(var_tensor.shape.dims) == 1
          and len(gamma_tensor.shape.dims) == 1
          and len(beta_tensor.shape.dims) == 1)
  # If the batch norm is after a FC layer, then the input/output tensors should
  # be in NC. Otherwise, the batch norm is after a convolution layer, and we
  # check backend_layouts for expected input/output layouts and do layout
  # transformation if needed.
  post_fc = False
  if len(input_tensor.shape.dims) == 2:
    post_fc = True

  if not post_fc:
    input_tensor = check_and_add_layout_transform(
        name=name, op=BatchNorm, input_tensors=[input_tensor])[0]

  mean_tensor.name = name + "/mean"
  var_tensor.name = name + "/var"
  gamma_tensor.name = name + "/gamma"
  beta_tensor.name = name + "/beta"
  output_layout = UnknownLayout
  output_layout = NC if post_fc else get_output_layout(BatchNorm)
  return add_node(
      name=name,
      op=BatchNorm,
      input_tensors=[
          input_tensor, mean_tensor, var_tensor, gamma_tensor, beta_tensor
      ],
      output_tensor_dims=input_tensor.shape.dims,
      output_tensor_layout=output_layout)

def max_pool(name, input_tensor, pool_size, stride):
  def compute_output_dim(input_dim, pool_size, stride):
    return (input_dim - pool_size) / stride + 1

  input_tensor = check_and_add_layout_transform(
      name=name, op=MaxPooling, input_tensors=[input_tensor])[0]

  row_idx = 2 if input_tensor.shape.layout == NCHW else 1
  col_idx = 3 if input_tensor.shape.layout == NCHW else 2
  output_rows = compute_output_dim(input_tensor.shape.dims[row_idx],
                                   pool_size[0], stride[0])
  output_cols = compute_output_dim(input_tensor.shape.dims[col_idx],
                                   pool_size[1], stride[1])
  output_layout = get_output_layout(MaxPooling)
  if output_layout == NCHW:
    output_tensor_dims = [
        input_tensor.shape.dims[0], input_tensor.shape.dims[1], output_rows,
        output_cols
    ]
  else:
    output_tensor_dims = [
        input_tensor.shape.dims[0], output_rows, output_cols,
        input_tensor.shape.dims[3]
    ]
  params = Params()
  params.pool_params.stride.extend(stride)
  params.pool_params.pool_size.extend(pool_size)
  return add_node(
      name=name,
      op=MaxPooling,
      input_tensors=[input_tensor],
      output_tensor_dims=output_tensor_dims,
      output_tensor_layout=output_layout,
      params=params)

def reorder(name, input_tensor, target_layout):
  src_layout = input_tensor.shape.layout
  src_dims = input_tensor.shape.dims
  if src_layout == NCHW:
    assert (target_layout == NHWC or target_layout == NC)
    if target_layout == NC:
      output_tensor_dims = [src_dims[0], np.prod(src_dims[1:])]
    else:
      output_tensor_dims = [src_dims[0], src_dims[2], src_dims[3], src_dims[1]]
  elif src_layout == NHWC:
    assert (target_layout == NCHW or target_layout == NC)
    if target_layout == NC:
      output_tensor_dims = [src_dims[0], np.prod(src_dims[1:])]
    else:
      output_tensor_dims = [src_dims[0], src_dims[3], src_dims[1], src_dims[2]]
  elif src_layout == NC:
    assert False, "Data layout reordering from NC is not supported!"

  return add_node(
      name=name,
      op=Reorder,
      input_tensors=[input_tensor],
      output_tensor_dims=output_tensor_dims,
      output_tensor_layout=target_layout)

def flatten(name, input_tensor):
  assert (len(input_tensor.shape.dims) == 4)
  return reorder(name=name, input_tensor=input_tensor, target_layout=NC)

def mat_mul(name, input_tensor, weight_tensor):
  weight_tensor.name = name + "/weights"

  input_tensor, weight_tensor = check_and_add_layout_transform(
      name=name, op=InnerProduct, input_tensors=[input_tensor, weight_tensor])

  assert (len(input_tensor.shape.dims) == 2
          and len(weight_tensor.shape.dims) == 2
          and input_tensor.shape.dims[1] == weight_tensor.shape.dims[0])
  output_tensor_dims = [input_tensor.shape.dims[0], weight_tensor.shape.dims[1]]
  return add_node(
      name=name,
      op=InnerProduct,
      input_tensors=[input_tensor, weight_tensor],
      output_tensor_dims=output_tensor_dims,
      output_tensor_layout=NC)

def add(name, tensor_a, tensor_b):
  assert (tensor_a.shape.dims == tensor_b.shape.dims
          ), "Elementwise add must have the same shape for the input tensors."
  return add_node(
      name=name,
      op=EltwiseAdd,
      input_tensors=[tensor_a, tensor_b],
      output_tensor_dims=tensor_a.shape.dims,
      output_tensor_layout=tensor_a.shape.layout)
