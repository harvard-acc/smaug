#!/usr/bin/env python

"""Examples for creating models.

This file contains two examples of creating models for SMAUG.
create_residual_model() creates a basic residual unit, and
create_sequential_model() creates a small sequential model.

Note that all the tensors and operators must be created within
a Graph context. The target backend is also specified via the
Graph context. See the below for examples. The user can get a
summary of the graph by calling print_summary(). To serialize
the graph into a pb file, call write_graph().
"""

import numpy as np
import smaug as sg

def create_residual_model():
  with sg.Graph(name="residual_graph", backend="SMV") as graph:
    # Tensors and kernels are initialized as NCHW layout.
    input_tensor = sg.Tensor(
        tensor_data=np.random.rand(1, 1, 28, 28).astype(np.float16))
    filter_tensor0 = sg.Tensor(
        tensor_data=np.random.rand(64, 1, 3, 3).astype(np.float16))
    filter_tensor1 = sg.Tensor(
        tensor_data=np.random.rand(64, 1, 3, 3).astype(np.float16))
    filter_tensor2 = sg.Tensor(
        tensor_data=np.random.rand(64, 64, 3, 3).astype(np.float16))
    bn_mean_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=np.random.rand(1, 64).astype(np.float16))
    bn_var_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=np.random.rand(1, 64).astype(np.float16))
    bn_gamma_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=np.random.rand(1, 64).astype(np.float16))
    bn_beta_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=np.random.rand(1, 64).astype(np.float16))

    act = sg.input_data(input_tensor)
    x = sg.nn.convolution(act, filter_tensor0, stride=[1, 1], padding="same")
    out = sg.nn.convolution(act, filter_tensor1, stride=[1, 1], padding="same")
    out = sg.nn.batch_norm(
        out, bn_mean_tensor, bn_var_tensor, bn_gamma_tensor, bn_beta_tensor,
        activation="relu")
    out = sg.nn.convolution(out, filter_tensor2, stride=[1, 1], padding="same")
    out = sg.math.add(x, out)

    return graph

def create_sequential_model():
  with sg.Graph(name="sequential_graph", backend="Reference") as graph:
    # Tensors and weights are initialized as NCHW layout.
    input_tensor = sg.Tensor(
        tensor_data=np.random.rand(1, 3, 32, 32).astype(np.float32))
    filter_tensor0 = sg.Tensor(
        tensor_data=np.random.rand(64, 3, 3, 3).astype(np.float32))
    filter_tensor1 = sg.Tensor(
        tensor_data=np.random.rand(64, 64, 3, 3).astype(np.float32))
    weight_tensor0 = sg.Tensor(
        data_layout=sg.NC, tensor_data=np.random.rand(256,
                                                      16384).astype(np.float32))
    weight_tensor1 = sg.Tensor(
        data_layout=sg.NC, tensor_data=np.random.rand(10,
                                                      256).astype(np.float32))
    bn_mean_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=np.random.rand(1, 64).astype(np.float32))
    bn_var_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=np.random.rand(1, 64).astype(np.float32))
    bn_gamma_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=np.random.rand(1, 64).astype(np.float32))
    bn_beta_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=np.random.rand(1, 64).astype(np.float32))

    out = sg.input_data(input_tensor)
    out = sg.nn.convolution(
        out, filter_tensor0, stride=[1, 1], padding="same", activation="relu")
    out = sg.nn.batch_norm(
        out, bn_mean_tensor, bn_var_tensor, bn_gamma_tensor, bn_beta_tensor)
    out = sg.nn.convolution(
        out, filter_tensor1, stride=[1, 1], padding="same", activation="relu")
    out = sg.nn.max_pool(out, pool_size=[2, 2], stride=[2, 2])
    out = sg.tensor.flatten(out)
    out = sg.nn.mat_mul(out, weight_tensor0, activation="relu")
    out = sg.nn.mat_mul(out, weight_tensor1)

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
