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
  if not get_graph().layout_trans_enabled:
    return input_tensors
  backend = get_graph().graph.backend
  for i in range(len(input_tensors)):
    expected_layoutset = backend_layouts[backend][op].input_layoutsets[i]
    input_layout = input_tensors[i].shape.layout
    if not expected_layoutset.contains(input_layout):
      input_tensors[i] = reorder(input_tensors[i], expected_layoutset.layouts)
  return input_tensors

def add_node(
    op, input_tensors, output_tensor_dims, output_tensor_layout=NCHW,
    output_tensor_dtype=None, output_tensor_dformat=Uncompressed, name=None,
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
          op=Data, input_tensors=[input_tensors[i]],
          output_tensor_dims=input_tensors[i].shape.dims,
          output_tensor_layout=input_tensors[i].shape.layout,
          output_tensor_dtype=output_tensor_dtype,
          output_tensor_dformat=output_tensor_dformat)
  return get_graph().add_node(
      name=name, op=op, input_tensors=input_tensors,
      output_tensor_dims=output_tensor_dims,
      output_tensor_layout=output_tensor_layout,
      output_tensor_dtype=output_tensor_dtype,
      output_tensor_dformat=output_tensor_dformat, params=params)

def input_data(input_tensor, name=None):
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

def set_activation_params(activation, act_params_proto, act_params):
  if activation not in supported_activations:
    raise AssertionError(
        ("%s is not a supported activation function. "
         "Supported activations: %s.")
        % (OpType.Name(activation),
           [OpType.Name(a) for a in supported_activations]))
  if act_params != None:
    if activation == LReLU:
      act_params_proto.lrelu_params.CopyFrom(act_params)
    elif activation == ELU:
      act_params_proto.elu_params.CopyFrom(act_params)
    elif activation == SELU:
      act_params_proto.elu_params.CopyFrom(act_params)
    elif activation == HardTanh:
      act_params_proto.hard_tanh_params.CopyFrom(act_params)
  else:
    # Use default values for the parameters if not specified.
    if activation == LReLU:
      act_params_proto.lrelu_params.slope = 0.2
    elif activation == ELU:
      act_params_proto.elu_params.alpha = 0.1
    elif activation == SELU:
      act_params_proto.elu_params.alpha = 1.6733
      act_params_proto.elu_params.lambda_param = 1.0507
    elif activation == HardTanh:
      act_params_proto.hard_tanh_params.min = -1
      act_params_proto.hard_tanh_params.max = 1

def convolution(
    input_tensor, filter_tensor, stride, padding, activation=None,
    activation_params=None, name=None):
  def compute_output_dim(input_dim, weight_dim, stride, padding):
    pad = 0
    if to_padding_type(padding) == SamePadding:
      pad = weight_dim - 1
    return (input_dim - weight_dim + pad) / stride + 1

  input_tensor, filter_tensor = check_and_add_layout_transform(
      name=name, op=Convolution3d, input_tensors=[input_tensor, filter_tensor])

  row_idx = 2 if input_tensor.shape.layout == NCHW else 1
  col_idx = 3 if input_tensor.shape.layout == NCHW else 2
  chan_idx = 1 if input_tensor.shape.layout == NCHW else 3
  assert input_tensor.dims(chan_idx) == filter_tensor.dims(chan_idx), (
      "The weights must have the same number of channels as the inputs.")
  output_rows = compute_output_dim(input_tensor.shape.dims[row_idx],
                                   filter_tensor.shape.dims[row_idx], stride[0],
                                   padding)
  output_cols = compute_output_dim(input_tensor.shape.dims[col_idx],
                                   filter_tensor.shape.dims[col_idx], stride[1],
                                   padding)
  output_layout = input_tensor.shape.layout
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
  if activation != None:
    params.act_params.activation = activation
    set_activation_params(activation, params.act_params, activation_params)

  return add_node(
      name=name,
      op=Convolution3d,
      input_tensors=[input_tensor, filter_tensor],
      output_tensor_dims=output_tensor_dims,
      output_tensor_layout=output_layout,
      params=params)

def relu(input_tensor, name=None):
  return add_node(
      name=name,
      op=ReLU,
      input_tensors=[input_tensor],
      output_tensor_dims=input_tensor.shape.dims,
      output_tensor_layout=input_tensor.shape.layout)

def lrelu(input_tensor, slope=0.2, name=None):
  params = Params()
  params.act_params.lrelu_params.slope = slope
  return add_node(
      name=name,
      op=LReLU,
      input_tensors=[input_tensor],
      output_tensor_dims=input_tensor.shape.dims,
      output_tensor_layout=input_tensor.shape.layout,
      params=params)

def elu(input_tensor, alpha=0.1, name=None):
  params = Params()
  params.act_params.elu_params.alpha = alpha
  return add_node(
      name=name,
      op=ELU,
      input_tensors=[input_tensor],
      output_tensor_dims=input_tensor.shape.dims,
      output_tensor_layout=input_tensor.shape.layout,
      params=params)

def selu(input_tensor, alpha=1.6733, lambda_param=1.0507, name=None):
  params = Params()
  params.act_params.elu_params.alpha = alpha
  params.act_params.elu_params.lambda_param = lambda_param
  return add_node(
      name=name,
      op=SELU,
      input_tensors=[input_tensor],
      output_tensor_dims=input_tensor.shape.dims,
      output_tensor_layout=input_tensor.shape.layout,
      params=params)

def tanh(input_tensor, name=None):
  return add_node(
      name=name,
      op=Tanh,
      input_tensors=[input_tensor],
      output_tensor_dims=input_tensor.shape.dims,
      output_tensor_layout=input_tensor.shape.layout)

def hard_tanh(input_tensor, min=-1, max=1, name=None):
  params = Params()
  params.act_params.hard_tanh_params.min = min
  params.act_params.hard_tanh_params.max = max
  return add_node(
      name=name,
      op=HardTanh,
      input_tensors=[input_tensor],
      output_tensor_dims=input_tensor.shape.dims,
      output_tensor_layout=input_tensor.shape.layout,
      params=params)

def sigmoid(input_tensor, name=None):
  return add_node(
      name=name,
      op=Sigmoid,
      input_tensors=[input_tensor],
      output_tensor_dims=input_tensor.shape.dims,
      output_tensor_layout=input_tensor.shape.layout)

def batch_norm(
    input_tensor, mean_tensor, var_tensor, gamma_tensor, beta_tensor,
    activation=None, activation_params=None, name=None):
  assert (len(mean_tensor.shape.dims) == 2 and len(var_tensor.shape.dims) == 2
          and len(gamma_tensor.shape.dims) == 2
          and len(beta_tensor.shape.dims) == 2)
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

  output_layout = UnknownLayout
  output_layout = NC if post_fc else input_tensor.shape.layout
  params = Params()
  if activation != None:
    params.act_params.activation = activation
    set_activation_params(activation, params.act_params, activation_params)
  return add_node(
      name=name,
      op=BatchNorm,
      input_tensors=[
          input_tensor, mean_tensor, var_tensor, gamma_tensor, beta_tensor
      ],
      output_tensor_dims=input_tensor.shape.dims,
      output_tensor_layout=output_layout,
      params=params)

def max_pool(input_tensor, pool_size, stride, name=None):
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
  output_layout = input_tensor.shape.layout
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

def reorder(input_tensor, target_layout, name=None):
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
  elif (src_layout == NC and target_layout == CN) or (src_layout == CN
                                                      and target_layout == NC):
    # 2D tensor transposition.
    output_tensor_dims = [src_dims[1], src_dims[0]]

  return add_node(
      name=name,
      op=Reorder,
      input_tensors=[input_tensor],
      output_tensor_dims=output_tensor_dims,
      output_tensor_layout=target_layout)

def flatten(input_tensor, name=None):
  assert (len(input_tensor.shape.dims) == 4)
  return reorder(name=name, input_tensor=input_tensor, target_layout=NC)

def mat_mul(
    input_tensor, weight_tensor, activation=None, activation_params=None,
    name=None):
  input_tensor, weight_tensor = check_and_add_layout_transform(
      name=name, op=InnerProduct, input_tensors=[input_tensor, weight_tensor])

  weight_layout = weight_tensor.shape.layout
  actIdx = 1 if weight_layout == NC else 0
  neuronIdx = 0 if weight_layout == NC else 1
  assert (len(input_tensor.shape.dims) == 2
          and len(weight_tensor.shape.dims) == 2
          and input_tensor.shape.dims[1] == weight_tensor.shape.dims[actIdx])
  output_tensor_dims = [
      input_tensor.shape.dims[0], weight_tensor.shape.dims[neuronIdx]
  ]
  params = Params()
  if activation != None:
    params.act_params.activation = activation
    set_activation_params(activation, params.act_params, activation_params)
  return add_node(
      name=name,
      op=InnerProduct,
      input_tensors=[input_tensor, weight_tensor],
      output_tensor_dims=output_tensor_dims,
      output_tensor_layout=NC,
      params=params)

def add(tensor_a, tensor_b, name=None):
  assert (tensor_a.shape.dims == tensor_b.shape.dims
          ), "Elementwise add must have the same shape for the input tensors."
  return add_node(
      name=name,
      op=EltwiseAdd,
      input_tensors=[tensor_a, tensor_b],
      output_tensor_dims=tensor_a.shape.dims,
      output_tensor_layout=tensor_a.shape.layout)

def mul(tensor_a, tensor_b, name=None):
  if tensor_a.shape.dims != tensor_b.shape.dims:
    raise ValueError(
        "Elementwise multiplication must have the same shape for the inputs!")
  return add_node(
      name=name,
      op=EltwiseMul,
      input_tensors=[tensor_a, tensor_b],
      output_tensor_dims=tensor_a.shape.dims,
      output_tensor_layout=tensor_a.shape.layout)
