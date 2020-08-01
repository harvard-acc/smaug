#!/usr/bin/env python
#
# Examples for creating models.
#
# This file contains two examples of creating models for SMAUG.
# create_residual_model() creates a basic residual unit, and
# create_sequential_model() creates a small sequential model.
#
# Note that all the tensors and operators must be created within
# a Graph context. The target backend is also specified via the
# Graph context. See the below for examples. The user can get a
# summary of the graph by calling print_summary(). To serialize
# the graph into a pb file, call write_graph().


import numpy as np
from smaug.core import types_pb2
from smaug.python.graph import Graph
from smaug.python.tensor import Tensor
from smaug.python.ops import array_ops, math_ops, data_op, nn_ops

def create_residual_model():
  with Graph(name="residual_graph", backend="SMV") as graph:
    # Tensors and kernels are initialized as NCHW layout.
    input_tensor = Tensor(
        tensor_data=np.random.rand(1, 1, 28, 28).astype(np.float16))
    filter_tensor0 = Tensor(
        tensor_data=np.random.rand(64, 1, 3, 3).astype(np.float16))
    filter_tensor1 = Tensor(
        tensor_data=np.random.rand(64, 1, 3, 3).astype(np.float16))
    filter_tensor2 = Tensor(
        tensor_data=np.random.rand(64, 64, 3, 3).astype(np.float16))
    bn_mean_tensor = Tensor(
        data_layout=types_pb2.NC,
        tensor_data=np.random.rand(1, 64).astype(np.float16))
    bn_var_tensor = Tensor(
        data_layout=types_pb2.NC,
        tensor_data=np.random.rand(1, 64).astype(np.float16))
    bn_gamma_tensor = Tensor(
        data_layout=types_pb2.NC,
        tensor_data=np.random.rand(1, 64).astype(np.float16))
    bn_beta_tensor = Tensor(
        data_layout=types_pb2.NC,
        tensor_data=np.random.rand(1, 64).astype(np.float16))

    act = data_op.input_data(input_tensor)
    x = nn_ops.convolution(act, filter_tensor0, stride=[1, 1], padding="same")
    out = nn_ops.convolution(act, filter_tensor1, stride=[1, 1], padding="same")
    out = nn_ops.batch_norm(
        out, bn_mean_tensor, bn_var_tensor, bn_gamma_tensor, bn_beta_tensor,
        activation=types_pb2.ReLU)
    out = nn_ops.convolution(out, filter_tensor2, stride=[1, 1], padding="same")
    out = math_ops.add(x, out)

    return graph


def create_sequential_model():
  with Graph(name="sequential_graph", backend="Reference") as graph:
    # Tensors and weights are initialized as NCHW layout.
    input_tensor = Tensor(
        tensor_data=np.random.rand(1, 3, 32, 32).astype(np.float32))
    filter_tensor0 = Tensor(
        tensor_data=np.random.rand(64, 3, 3, 3).astype(np.float32))
    filter_tensor1 = Tensor(
        tensor_data=np.random.rand(64, 64, 3, 3).astype(np.float32))
    weight_tensor0 = Tensor(
        data_layout=types_pb2.NC,
        tensor_data=np.random.rand(256, 16384).astype(np.float32))
    weight_tensor1 = Tensor(
        data_layout=types_pb2.NC,
        tensor_data=np.random.rand(10, 256).astype(np.float32))
    bn_mean_tensor = Tensor(
        data_layout=types_pb2.NC,
        tensor_data=np.random.rand(1, 64).astype(np.float32))
    bn_var_tensor = Tensor(
        data_layout=types_pb2.NC,
        tensor_data=np.random.rand(1, 64).astype(np.float32))
    bn_gamma_tensor = Tensor(
        data_layout=types_pb2.NC,
        tensor_data=np.random.rand(1, 64).astype(np.float32))
    bn_beta_tensor = Tensor(
        data_layout=types_pb2.NC,
        tensor_data=np.random.rand(1, 64).astype(np.float32))

    out = data_op.input_data(input_tensor)
    out = nn_ops.convolution(
        out, filter_tensor0, stride=[1, 1], padding="same", activation=types_pb2.ReLU)
    out = nn_ops.batch_norm(
        out, bn_mean_tensor, bn_var_tensor, bn_gamma_tensor, bn_beta_tensor)
    out = nn_ops.convolution(
        out, filter_tensor1, stride=[1, 1], padding="same", activation=types_pb2.ReLU)
    out = nn_ops.max_pool(out, pool_size=[2, 2], stride=[2, 2])
    out = array_ops.flatten(out)
    out = nn_ops.mat_mul(out, weight_tensor0, activation=types_pb2.ReLU)
    out = nn_ops.mat_mul(out, weight_tensor1)

    return graph

if __name__ != "main":
  # Create a sequential model.
  seq_graph = create_sequential_model()

  # Print summary of the graph.
  seq_graph.print_summary()

  # Write the graph to file.
  seq_graph.write_graph()

  # Create a residual model.
  res_graph = create_residual_model()

  # Print summary of the graph.
  res_graph.print_summary()

  # Write the graph to file.
  res_graph.write_graph()
