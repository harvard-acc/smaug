import numpy as np
from types_pb2 import *
from node_pb2 import *
from global_vars import *
from tensor import *

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
      output_tensor_layout=NCHW)

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
  output_tensor_dims = [
      input_tensor.shape.dims[0],
      filter_tensor.shape.dims[0],
      compute_output_dim(input_tensor.shape.dims[2],
                         filter_tensor.shape.dims[2], stride[0], padding),
      compute_output_dim(input_tensor.shape.dims[3],
                         filter_tensor.shape.dims[3], stride[1], padding),
  ]
  params = Params()
  params.conv_params.padding = to_padding_type(padding)
  params.conv_params.stride.extend(stride)
  return add_node(
      name=name,
      op=Convolution3d,
      input_tensors=[input_tensor, filter_tensor],
      output_tensor_dims=output_tensor_dims,
      params=params)

def relu(name, input_tensor):
  return add_node(
      name=name,
      op=ReLU,
      input_tensors=[input_tensor],
      output_tensor_dims=input_tensor.shape.dims,
      output_tensor_layout=X)

def batch_norm(name, input_tensor, mean_tensor, var_tensor, gamma_tensor,
               beta_tensor):
  mean_tensor.name = name + "/mean"
  var_tensor.name = name + "/var"
  gamma_tensor.name = name + "/gamma"
  beta_tensor.name = name + "/beta"
  return add_node(
      name=name,
      op=BatchNorm,
      input_tensors=[
          input_tensor, mean_tensor, var_tensor, gamma_tensor, beta_tensor
      ],
      output_tensor_dims=input_tensor.shape.dims,
      output_tensor_layout=X)

def max_pool(name, input_tensor, pool_size, stride):
  def compute_output_dim(input_dim, pool_size, stride):
    return (input_dim - pool_size) / stride + 1

  output_tensor_dims = [
      input_tensor.shape.dims[0], input_tensor.shape.dims[1],
      compute_output_dim(input_tensor.shape.dims[2], pool_size[0], stride[0]),
      compute_output_dim(input_tensor.shape.dims[3], pool_size[1], stride[1])
  ]
  params = Params()
  params.pool_params.stride.extend(stride)
  params.pool_params.pool_size.extend(pool_size)
  return add_node(
      name=name,
      op=MaxPooling,
      input_tensors=[input_tensor],
      output_tensor_dims=output_tensor_dims,
      params=params)

def flatten(name, input_tensor):
  assert (len(input_tensor.shape.dims) == 4)
  output_tensor_dims = [
      input_tensor.shape.dims[0],
      np.prod(input_tensor.shape.dims[1:])
  ]
  return add_node(
      name=name,
      op=Reorder,
      input_tensors=[input_tensor],
      output_tensor_dims=output_tensor_dims,
      output_tensor_layout=NC)

def mat_mul(name, input_tensor, weight_tensor):
  assert (len(input_tensor.shape.dims) == 2
          and len(weight_tensor.shape.dims) == 2
          and input_tensor.shape.dims[1] == weight_tensor.shape.dims[0])
  weight_tensor.name = name + "/weights"
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
      output_tensor_layout=X)
