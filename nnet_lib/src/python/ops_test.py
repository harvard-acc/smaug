#!/usr/bin/env python

"""Tests for python/ops.py."""

import unittest
import smaug_test
from graph import *
from tensor import *
from ops import *
from types_pb2 import *

class OperationTest(smaug_test.SmaugTest):
  def __init__(self, *args, **kwargs):
    """Create a test graph for all the tests."""
    super(OperationTest, self).__init__(*args, **kwargs)
    self.build_test_graph()

  def build_test_graph(self):
    with Graph(name="test_graph", backend="SMV") as graph:
      input_tensor = Tensor(
          data_layout=NCHW,
          tensor_data=np.random.rand(1, 3, 28, 28).astype(np.float16))
      filter_tensor0 = Tensor(
          data_layout=NCHW,
          tensor_data=np.random.rand(64, 3, 3, 3).astype(np.float16))
      filter_tensor1 = Tensor(
          data_layout=NCHW,
          tensor_data=np.random.rand(64, 64, 3, 3).astype(np.float16))
      weight_tensor0 = Tensor(
          data_layout=NC,
          tensor_data=np.random.rand(12544, 254).astype(np.float16))
      weight_tensor1 = Tensor(
          data_layout=NC,
          tensor_data=np.random.rand(254, 10).astype(np.float16))
      bn_mean_tensor = Tensor(
          data_layout=X, tensor_data=np.random.rand(1, 64).astype(np.float16))
      bn_var_tensor = Tensor(
          data_layout=X, tensor_data=np.random.rand(1, 64).astype(np.float16))
      bn_gamma_tensor = Tensor(
          data_layout=X, tensor_data=np.random.rand(1, 64).astype(np.float16))
      bn_beta_tensor = Tensor(
          data_layout=X, tensor_data=np.random.rand(1, 64).astype(np.float16))

      out = input_data("input", input_tensor)
      out = convolution(
          "conv0", out, filter_tensor0, stride=[1, 1], padding="same")
      out = relu("conv0_relu", out)
      out = batch_norm("bn", out, bn_mean_tensor, bn_var_tensor,
                       bn_gamma_tensor, bn_beta_tensor)
      out = convolution(
          "conv1", out, filter_tensor1, stride=[1, 1], padding="same")
      out = relu("conv1_relu", out)
      out = max_pool("pool", out, pool_size=[2, 2], stride=[2, 2])
      out = flatten("flatten", out)
      out = mat_mul("fc0", out, weight_tensor0)
      out = relu("fc0_relu", out)
      out = mat_mul("fc1", out, weight_tensor1)

    self.test_graph = graph
    self.alignment = backend_alignment[self.test_graph.graph.backend]

  def test_input_op(self):
    node = self.get_node(self.test_graph.graph, "input")
    self.assertEqual(node.op, Data)
    self.assertEqual(len(node.parents), 0)
    self.assertEqual(node.children[0], "conv0")
    self.assertEqual(len(node.input_tensors), 1)
    self.assertEqual(len(node.output_tensors), 1)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "input")
    self.assertEqual(node.output_tensors[0].data_type, Float16)
    self.assertEqual(node.output_tensors[0].shape.dims, [1, 3, 28, 28])
    self.assertEqual(node.output_tensors[0].shape.layout, NCHW)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

  def test_convolution_op(self):
    # The first convolution operator "conv0"
    node = self.get_node(self.test_graph.graph, "conv0")
    self.assertEqual(node.op, Convolution3d)
    self.assertEqual(node.parents[0], "input")
    self.assertEqual(node.children[0], "conv0_relu")
    self.assertEqual(len(node.input_tensors), 2)
    self.assertEqual(len(node.output_tensors), 1)
    # Parameters
    self.assertEqual(node.params.conv_params.padding, SamePadding)
    self.assertEqual(node.params.conv_params.stride, [1, 1])
    # Filter tensor
    self.assertEqual(node.input_tensors[1].name, "conv0/kernels")
    self.assertEqual(node.input_tensors[1].data_type, Float16)
    self.assertEqual(node.input_tensors[1].shape.dims, [64, 3, 3, 3])
    self.assertEqual(node.input_tensors[1].shape.layout, NCHW)
    self.assertEqual(node.input_tensors[1].shape.alignment, self.alignment)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "conv0")
    self.assertEqual(node.output_tensors[0].data_type, Float16)
    self.assertEqual(node.output_tensors[0].shape.dims, [1, 64, 28, 28])
    self.assertEqual(node.output_tensors[0].shape.layout, NCHW)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

    # The second convolution operator "conv1"
    node = self.get_node(self.test_graph.graph, "conv1")
    self.assertEqual(node.op, Convolution3d)
    self.assertEqual(node.parents[0], "bn")
    self.assertEqual(node.children[0], "conv1_relu")
    self.assertEqual(len(node.input_tensors), 2)
    self.assertEqual(len(node.output_tensors), 1)
    # Parameters
    self.assertEqual(node.params.conv_params.padding, SamePadding)
    self.assertEqual(node.params.conv_params.stride, [1, 1])
    # Filter tensor
    self.assertEqual(node.input_tensors[1].name, "conv1/kernels")
    self.assertEqual(node.input_tensors[1].data_type, Float16)
    self.assertEqual(node.input_tensors[1].shape.dims, [64, 64, 3, 3])
    self.assertEqual(node.input_tensors[1].shape.layout, NCHW)
    self.assertEqual(node.input_tensors[1].shape.alignment, self.alignment)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "conv1")
    self.assertEqual(node.output_tensors[0].data_type, Float16)
    self.assertEqual(node.output_tensors[0].shape.dims, [1, 64, 28, 28])
    self.assertEqual(node.output_tensors[0].shape.layout, NCHW)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

  def test_relu_op(self):
    # The first relu operator "conv0_relu"
    node = self.get_node(self.test_graph.graph, "conv0_relu")
    self.assertEqual(node.op, ReLU)
    self.assertEqual(node.parents[0], "conv0")
    self.assertEqual(node.children[0], "bn")
    self.assertEqual(len(node.input_tensors), 1)
    self.assertEqual(len(node.output_tensors), 1)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "conv0_relu")
    self.assertEqual(node.output_tensors[0].data_type, Float16)
    self.assertEqual(node.output_tensors[0].shape.dims, [1, 64, 28, 28])
    self.assertEqual(node.output_tensors[0].shape.layout, X)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

    # The second relu operator "conv1_relu"
    node = self.get_node(self.test_graph.graph, "conv1_relu")
    self.assertEqual(node.op, ReLU)
    self.assertEqual(node.parents[0], "conv1")
    self.assertEqual(node.children[0], "pool")
    self.assertEqual(len(node.input_tensors), 1)
    self.assertEqual(len(node.output_tensors), 1)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "conv1_relu")
    self.assertEqual(node.output_tensors[0].data_type, Float16)
    self.assertEqual(node.output_tensors[0].shape.dims, [1, 64, 28, 28])
    self.assertEqual(node.output_tensors[0].shape.layout, X)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

    # The third relu operator "fc0_relu"
    node = self.get_node(self.test_graph.graph, "fc0_relu")
    self.assertEqual(node.op, ReLU)
    self.assertEqual(node.parents[0], "fc0")
    self.assertEqual(node.children[0], "fc1")
    self.assertEqual(len(node.input_tensors), 1)
    self.assertEqual(len(node.output_tensors), 1)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "fc0_relu")
    self.assertEqual(node.output_tensors[0].data_type, Float16)
    self.assertEqual(node.output_tensors[0].shape.dims, [1, 254])
    self.assertEqual(node.output_tensors[0].shape.layout, X)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

  def test_batch_norm_op(self):
    node = self.get_node(self.test_graph.graph, "bn")
    self.assertEqual(node.op, BatchNorm)
    self.assertEqual(node.parents[0], "conv0_relu")
    self.assertEqual(node.children[0], "conv1")
    self.assertEqual(len(node.input_tensors), 5)
    self.assertEqual(len(node.output_tensors), 1)
    # Weight tensors
    self.assertEqual(node.input_tensors[1].name, "bn/mean")
    self.assertEqual(node.input_tensors[1].data_type, Float16)
    self.assertEqual(node.input_tensors[1].shape.dims, [1, 64])
    self.assertEqual(node.input_tensors[1].shape.layout, X)
    self.assertEqual(node.input_tensors[1].shape.alignment, self.alignment)
    self.assertEqual(node.input_tensors[2].name, "bn/var")
    self.assertEqual(node.input_tensors[2].data_type, Float16)
    self.assertEqual(node.input_tensors[2].shape.dims, [1, 64])
    self.assertEqual(node.input_tensors[2].shape.layout, X)
    self.assertEqual(node.input_tensors[2].shape.alignment, self.alignment)
    self.assertEqual(node.input_tensors[3].name, "bn/gamma")
    self.assertEqual(node.input_tensors[3].data_type, Float16)
    self.assertEqual(node.input_tensors[3].shape.dims, [1, 64])
    self.assertEqual(node.input_tensors[3].shape.layout, X)
    self.assertEqual(node.input_tensors[3].shape.alignment, self.alignment)
    self.assertEqual(node.input_tensors[4].name, "bn/beta")
    self.assertEqual(node.input_tensors[4].data_type, Float16)
    self.assertEqual(node.input_tensors[4].shape.dims, [1, 64])
    self.assertEqual(node.input_tensors[4].shape.layout, X)
    self.assertEqual(node.input_tensors[4].shape.alignment, self.alignment)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "bn")
    self.assertEqual(node.output_tensors[0].data_type, Float16)
    self.assertEqual(node.output_tensors[0].shape.dims, [1, 64, 28, 28])
    self.assertEqual(node.output_tensors[0].shape.layout, X)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

  def test_max_pool_op(self):
    node = self.get_node(self.test_graph.graph, "pool")
    self.assertEqual(node.op, MaxPooling)
    self.assertEqual(node.parents[0], "conv1_relu")
    self.assertEqual(node.children[0], "flatten")
    self.assertEqual(len(node.input_tensors), 1)
    self.assertEqual(len(node.output_tensors), 1)
    # Parameters
    self.assertEqual(node.params.pool_params.stride, [2, 2])
    self.assertEqual(node.params.pool_params.pool_size, [2, 2])
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "pool")
    self.assertEqual(node.output_tensors[0].data_type, Float16)
    self.assertEqual(node.output_tensors[0].shape.dims, [1, 64, 14, 14])
    self.assertEqual(node.output_tensors[0].shape.layout, NCHW)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

  def test_flatten_op(self):
    node = self.get_node(self.test_graph.graph, "flatten")
    self.assertEqual(node.op, Reorder)
    self.assertEqual(node.parents[0], "pool")
    self.assertEqual(node.children[0], "fc0")
    self.assertEqual(len(node.input_tensors), 1)
    self.assertEqual(len(node.output_tensors), 1)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "flatten")
    self.assertEqual(node.output_tensors[0].data_type, Float16)
    self.assertEqual(node.output_tensors[0].shape.dims, [1, 12544])
    self.assertEqual(node.output_tensors[0].shape.layout, NC)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

  def test_mat_mul_op(self):
    # The first mat_mul operator "fc0"
    node = self.get_node(self.test_graph.graph, "fc0")
    self.assertEqual(node.op, InnerProduct)
    self.assertEqual(node.parents[0], "flatten")
    self.assertEqual(node.children[0], "fc0_relu")
    self.assertEqual(len(node.input_tensors), 2)
    self.assertEqual(len(node.output_tensors), 1)
    # Weight tensor
    self.assertEqual(node.input_tensors[1].name, "fc0/weights")
    self.assertEqual(node.input_tensors[1].data_type, Float16)
    self.assertEqual(node.input_tensors[1].shape.dims, [12544, 254])
    self.assertEqual(node.input_tensors[1].shape.layout, NC)
    self.assertEqual(node.input_tensors[1].shape.alignment, self.alignment)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "fc0")
    self.assertEqual(node.output_tensors[0].data_type, Float16)
    self.assertEqual(node.output_tensors[0].shape.dims, [1, 254])
    self.assertEqual(node.output_tensors[0].shape.layout, NC)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

    # The second mat_mul operator "fc1"
    node = self.get_node(self.test_graph.graph, "fc1")
    self.assertEqual(node.op, InnerProduct)
    self.assertEqual(node.parents[0], "fc0_relu")
    self.assertEqual(len(node.children), 0)
    self.assertEqual(len(node.input_tensors), 2)
    self.assertEqual(len(node.output_tensors), 1)
    # Weight tensor
    self.assertEqual(node.input_tensors[1].name, "fc1/weights")
    self.assertEqual(node.input_tensors[1].data_type, Float16)
    self.assertEqual(node.input_tensors[1].shape.dims, [254, 10])
    self.assertEqual(node.input_tensors[1].shape.layout, NC)
    self.assertEqual(node.input_tensors[1].shape.alignment, self.alignment)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "fc1")
    self.assertEqual(node.output_tensors[0].data_type, Float16)
    self.assertEqual(node.output_tensors[0].shape.dims, [1, 10])
    self.assertEqual(node.output_tensors[0].shape.layout, NC)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

if __name__ == "__main__":
  unittest.main()
