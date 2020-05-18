#!/usr/bin/env python

"""Tests for python/ops.py."""

import unittest

from smaug.python import smaug_test
from smaug.python.graph import Graph
from smaug.python.tensor import Tensor
from smaug.python.ops.data_op import *
from smaug.python.ops.activation_ops import *
from smaug.python.ops.array_ops import *
from smaug.python.ops.nn_ops import *
from smaug.python.ops.math_ops import *
from smaug.python.datatypes import *
from smaug.core.types_pb2 import *

test_backend_dtypes = {"Reference": np.float32, "SMV": np.float16}

class OperatorTest:
  def assertEqualDims(self, dims, layout, expected_dims, expected_layout):
    """Test equality between two dims.

    Test if the two dims share the same dimensions or are merely one single
    shape in different layouts.

    Args:
      dims: A list of dimensions.
      layout: The layout of dims.
      expected_dims: A list of dimensions in the expected layout.
      expected_layout: The expected layout.
    """
    if layout == expected_layout:
      self.assertEqual(dims, expected_dims)
    elif layout == NHWC:
      assert expected_layout == NCHW
      self.assertEqual([dims[0], dims[3], dims[1], dims[2]], expected_dims)
    elif layout == NCHW:
      assert expected_layout == NHWC
      self.assertEqual([dims[0], dims[2], dims[3], dims[1]], expected_dims)
    elif layout == NC or layout == CN:
      assert len(expected_dims) == 2
      self.assertEqual([dims[1], dims[0]], expected_dims)
    else:
      assert False, "Other layouts not expected here!"

  def build_test_sequential_graph(self, backend):
    """Create a sequential model."""
    np_dtype = test_backend_dtypes[backend]
    self.expected_dtype = np_to_smaug_type[np_dtype]
    with Graph(name="test_sequential_graph", backend=backend) as graph:
      input_tensor = Tensor(
          data_layout=NCHW,
          tensor_data=np.random.rand(1, 3, 28, 28).astype(np_dtype))
      filter_tensor0 = Tensor(
          data_layout=NCHW,
          tensor_data=np.random.rand(64, 3, 3, 3).astype(np_dtype))
      filter_tensor1 = Tensor(
          data_layout=NCHW,
          tensor_data=np.random.rand(64, 64, 3, 3).astype(np_dtype))
      weight_tensor0 = Tensor(
          data_layout=NC,
          tensor_data=np.random.rand(254, 12544).astype(np_dtype))
      weight_tensor1 = Tensor(
          data_layout=NC, tensor_data=np.random.rand(10, 254).astype(np_dtype))
      bn_mean_tensor = Tensor(
          data_layout=NC, tensor_data=np.random.rand(1, 64).astype(np_dtype))
      bn_var_tensor = Tensor(
          data_layout=NC, tensor_data=np.random.rand(1, 64).astype(np_dtype))
      bn_gamma_tensor = Tensor(
          data_layout=NC, tensor_data=np.random.rand(1, 64).astype(np_dtype))
      bn_beta_tensor = Tensor(
          data_layout=NC, tensor_data=np.random.rand(1, 64).astype(np_dtype))

      out = input_data(input_tensor, "input")
      out = convolution(
          out, filter_tensor0, stride=[1, 1], padding="same", name="conv0")
      out = relu(out, "conv0_relu")
      out = batch_norm(
          out, bn_mean_tensor, bn_var_tensor, bn_gamma_tensor, bn_beta_tensor,
          name="bn")
      out = convolution(
          out, filter_tensor1, stride=[1, 1], padding="same", name="conv1")
      out = relu(out, "conv1_relu")
      out = max_pool(out, pool_size=[2, 2], stride=[2, 2], name="pool")
      out = flatten(out, "flatten")
      out = mat_mul(out, weight_tensor0, name="fc0")
      out = relu(out, "fc0_relu")
      out = mat_mul(out, weight_tensor1, name="fc1")
      out = expand_dims(out, 1, "expand_dims")
      out = squeeze(out, 1, "squeeze")
      out = reshape(out, [2, 5], NC, "reshape")
      out = repeat(out, [4, 2], "repeat")
      out = stack(out, 4, 1, "stack")
      out0, out1, out2, out3 = unstack(out, 1, "unstack")

    self.test_graph = graph
    self.backend = backend
    self.alignment = backend_alignment[backend]

  def build_test_residual_graph(self, backend):
    """Create a residual model.

    The graph contains a residual connection, where the output of conv0 and
    conv2 is added at the end."""

    np_dtype = test_backend_dtypes[backend]
    self.expected_dtype = np_to_smaug_type[np_dtype]
    with Graph(name="test_residual_graph", backend=backend) as graph:
      input_tensor = Tensor(
          data_layout=NCHW,
          tensor_data=np.random.rand(1, 1, 28, 28).astype(np_dtype))
      filter_tensor0 = Tensor(
          data_layout=NCHW,
          tensor_data=np.random.rand(64, 1, 3, 3).astype(np_dtype))
      filter_tensor1 = Tensor(
          data_layout=NCHW,
          tensor_data=np.random.rand(64, 1, 3, 3).astype(np_dtype))
      filter_tensor2 = Tensor(
          data_layout=NCHW,
          tensor_data=np.random.rand(64, 64, 3, 3).astype(np_dtype))
      bn_mean_tensor = Tensor(
          data_layout=NC, tensor_data=np.random.rand(1, 64).astype(np_dtype))
      bn_var_tensor = Tensor(
          data_layout=NC, tensor_data=np.random.rand(1, 64).astype(np_dtype))
      bn_gamma_tensor = Tensor(
          data_layout=NC, tensor_data=np.random.rand(1, 64).astype(np_dtype))
      bn_beta_tensor = Tensor(
          data_layout=NC, tensor_data=np.random.rand(1, 64).astype(np_dtype))

      act = input_data(input_tensor, "input")
      x = convolution(
          act, filter_tensor0, stride=[1, 1], padding="same", name="conv0")
      out = convolution(
          act, filter_tensor1, stride=[1, 1], padding="same", name="conv1")
      out = batch_norm(
          out, bn_mean_tensor, bn_var_tensor, bn_gamma_tensor, bn_beta_tensor,
          name="bn")
      out = relu(out, "relu")
      out = convolution(
          out, filter_tensor2, stride=[1, 1], padding="same", name="conv2")
      out = add(x, out, "add")
      out = mul(x, out, "mul")
      # Concatenate the channel dimension of x and out.
      axis = 1 if out.shape.layout == NCHW else 3
      out = concat([x, out], axis, "concat")
      # Evenly split the tensor into 4 over the channel dimension.
      out0, out1, out2, out3 = split(out, 4, axis, "split")
      out = mul(add(out0, out1, "add1"), add(out2, out3, "add2"), "mul1")

    self.test_graph = graph
    self.backend = backend
    self.alignment = backend_alignment[self.test_graph.graph.backend]

class SequentialGraphTest(OperatorTest):
  """Common tests for the sequential graph."""

  def test_input_op(self):
    node = self.get_node(self.test_graph.graph, "input")
    self.assertEqual(node.op, Data)
    self.assertEqual(len(node.input_tensors), 1)
    self.assertEqual(len(node.output_tensors), 1)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "input/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqual(node.output_tensors[0].shape.dims, [1, 3, 28, 28])
    self.assertEqual(node.output_tensors[0].shape.layout, NCHW)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

  def test_convolution_op(self):
    expected_weight_layout = backend_layouts[
        self.backend][Convolution3d].input_layoutsets[1].layouts
    expected_output_layout = backend_layouts[
        self.backend][Convolution3d].output_layoutset.layouts
    # The first convolution operator "conv0"
    node = self.get_node(self.test_graph.graph, "conv0")
    self.assertEqual(node.op, Convolution3d)
    self.assertEqual(len(node.input_tensors), 2)
    self.assertEqual(len(node.output_tensors), 1)
    # Parameters
    self.assertEqual(node.params.conv_params.padding, SamePadding)
    self.assertEqual(node.params.conv_params.stride, [1, 1])
    # Weight tensor
    self.assertEqual(node.input_tensors[1].data_type, self.expected_dtype)
    self.assertEqualDims(node.input_tensors[1].shape.dims,
                         node.input_tensors[1].shape.layout, [64, 3, 3, 3],
                         NCHW)
    self.assertEqual(node.input_tensors[1].shape.layout, expected_weight_layout)
    self.assertEqual(node.input_tensors[1].shape.alignment, self.alignment)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "conv0/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqualDims(node.output_tensors[0].shape.dims,
                         node.output_tensors[0].shape.layout, [1, 64, 28, 28],
                         NCHW)
    self.assertEqual(node.output_tensors[0].shape.layout,
                     expected_output_layout)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

    # The second convolution operator "conv1"
    node = self.get_node(self.test_graph.graph, "conv1")
    self.assertEqual(node.op, Convolution3d)
    self.assertEqual(len(node.input_tensors), 2)
    self.assertEqual(len(node.output_tensors), 1)
    # Parameters
    self.assertEqual(node.params.conv_params.padding, SamePadding)
    self.assertEqual(node.params.conv_params.stride, [1, 1])
    # Weight tensor
    self.assertEqual(node.input_tensors[1].data_type, self.expected_dtype)
    self.assertEqualDims(node.input_tensors[1].shape.dims,
                         node.input_tensors[1].shape.layout, [64, 64, 3, 3],
                         NCHW)
    self.assertEqual(node.input_tensors[1].shape.layout, expected_weight_layout)
    self.assertEqual(node.input_tensors[1].shape.alignment, self.alignment)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "conv1/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqualDims(node.output_tensors[0].shape.dims,
                         node.output_tensors[0].shape.layout, [1, 64, 28, 28],
                         NCHW)
    self.assertEqual(node.output_tensors[0].shape.layout,
                     expected_output_layout)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

  def test_relu_op(self):
    # The first relu operator "conv0_relu"
    node = self.get_node(self.test_graph.graph, "conv0_relu")
    self.assertEqual(node.op, ReLU)
    self.assertEqual(len(node.input_tensors), 1)
    self.assertEqual(len(node.output_tensors), 1)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "conv0_relu/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqualDims(node.output_tensors[0].shape.dims,
                         node.output_tensors[0].shape.layout, [1, 64, 28, 28],
                         NCHW)
    self.assertEqual(node.output_tensors[0].shape.layout,
                     node.input_tensors[0].shape.layout)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

    # The second relu operator "conv1_relu"
    node = self.get_node(self.test_graph.graph, "conv1_relu")
    self.assertEqual(node.op, ReLU)
    self.assertEqual(len(node.input_tensors), 1)
    self.assertEqual(len(node.output_tensors), 1)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "conv1_relu/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqualDims(node.output_tensors[0].shape.dims,
                         node.output_tensors[0].shape.layout, [1, 64, 28, 28],
                         NCHW)
    self.assertEqual(node.output_tensors[0].shape.layout,
                     node.input_tensors[0].shape.layout)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

    # The third relu operator "fc0_relu"
    node = self.get_node(self.test_graph.graph, "fc0_relu")
    self.assertEqual(node.op, ReLU)
    self.assertEqual(len(node.input_tensors), 1)
    self.assertEqual(len(node.output_tensors), 1)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "fc0_relu/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqual(node.output_tensors[0].shape.dims, [1, 254])
    self.assertEqual(node.output_tensors[0].shape.layout,
                     node.input_tensors[0].shape.layout)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

  def test_batch_norm_op(self):
    node = self.get_node(self.test_graph.graph, "bn")
    self.assertEqual(node.op, BatchNorm)
    self.assertEqual(len(node.input_tensors), 5)
    self.assertEqual(len(node.output_tensors), 1)
    # Weight tensors
    for i in [1, 2, 3, 4]:
      self.assertEqual(node.input_tensors[i].data_type, self.expected_dtype)
      self.assertEqual(node.input_tensors[i].shape.dims, [1, 64])
      self.assertEqual(node.input_tensors[i].shape.layout, NC)
      self.assertEqual(node.input_tensors[i].shape.alignment, self.alignment)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "bn/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqualDims(node.output_tensors[0].shape.dims,
                         node.output_tensors[0].shape.layout, [1, 64, 28, 28],
                         NCHW)
    self.assertEqual(node.output_tensors[0].shape.layout,
                     node.input_tensors[0].shape.layout)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

  def test_max_pool_op(self):
    expected_output_layout = backend_layouts[
        self.backend][MaxPooling].output_layoutset.layouts
    node = self.get_node(self.test_graph.graph, "pool")
    self.assertEqual(node.op, MaxPooling)
    self.assertEqual(len(node.input_tensors), 1)
    self.assertEqual(len(node.output_tensors), 1)
    # Parameters
    self.assertEqual(node.params.pool_params.stride, [2, 2])
    self.assertEqual(node.params.pool_params.pool_size, [2, 2])
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "pool/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqualDims(node.output_tensors[0].shape.dims,
                         node.output_tensors[0].shape.layout, [1, 64, 14, 14],
                         NCHW)
    self.assertEqual(node.output_tensors[0].shape.layout,
                     expected_output_layout)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

  def test_flatten_op(self):
    node = self.get_node(self.test_graph.graph, "flatten")
    self.assertEqual(node.op, Reorder)
    self.assertEqual(len(node.input_tensors), 1)
    self.assertEqual(len(node.output_tensors), 1)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "flatten/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqual(node.output_tensors[0].shape.dims, [1, 12544])
    self.assertEqual(node.output_tensors[0].shape.layout, NC)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

  def test_mat_mul_op(self):
    expected_weight_layout = backend_layouts[
        self.backend][InnerProduct].input_layoutsets[1].layouts
    expected_output_layout = backend_layouts[
        self.backend][InnerProduct].output_layoutset.layouts
    # The first mat_mul operator "fc0"
    node = self.get_node(self.test_graph.graph, "fc0")
    self.assertEqual(node.op, InnerProduct)
    self.assertEqual(len(node.input_tensors), 2)
    self.assertEqual(len(node.output_tensors), 1)
    # Weight tensor
    self.assertEqual(node.input_tensors[1].data_type, self.expected_dtype)
    self.assertEqualDims(node.input_tensors[1].shape.dims,
                         node.input_tensors[1].shape.layout, [254, 12544], NC)
    self.assertEqual(node.input_tensors[1].shape.layout, expected_weight_layout)
    self.assertEqual(node.input_tensors[1].shape.alignment, self.alignment)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "fc0/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqualDims(node.output_tensors[0].shape.dims,
                         node.output_tensors[0].shape.layout, [1, 254], NC)
    self.assertEqual(node.output_tensors[0].shape.layout,
                     expected_output_layout)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

    # The second mat_mul operator "fc1"
    node = self.get_node(self.test_graph.graph, "fc1")
    self.assertEqual(node.op, InnerProduct)
    self.assertEqual(len(node.input_tensors), 2)
    self.assertEqual(len(node.output_tensors), 1)
    # Weight tensor
    self.assertEqual(node.input_tensors[1].data_type, self.expected_dtype)
    self.assertEqualDims(node.input_tensors[1].shape.dims,
                         node.input_tensors[1].shape.layout, [10, 254], NC)
    self.assertEqual(node.input_tensors[1].shape.layout, expected_weight_layout)
    self.assertEqual(node.input_tensors[1].shape.alignment, self.alignment)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "fc1/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqualDims(node.output_tensors[0].shape.dims,
                         node.output_tensors[0].shape.layout, [1, 10], NC)
    self.assertEqual(node.output_tensors[0].shape.layout,
                     expected_output_layout)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

  def test_expand_dims_op(self):
    node = self.get_node(self.test_graph.graph, "expand_dims")
    self.assertEqual(node.op, Reshape)
    self.assertEqual(len(node.input_tensors), 1)
    self.assertEqual(len(node.output_tensors), 1)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "expand_dims/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqual(node.output_tensors[0].shape.dims, [1, 1, 10])
    self.assertEqual(node.output_tensors[0].shape.layout, NTC)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

  def test_squeeze_op(self):
    node = self.get_node(self.test_graph.graph, "squeeze")
    self.assertEqual(node.op, Reshape)
    self.assertEqual(len(node.input_tensors), 1)
    self.assertEqual(len(node.output_tensors), 1)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "squeeze/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqual(node.output_tensors[0].shape.dims, [1, 10])
    self.assertEqual(node.output_tensors[0].shape.layout, NC)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

  def test_reshape_op(self):
    node = self.get_node(self.test_graph.graph, "reshape")
    self.assertEqual(node.op, Reshape)
    self.assertEqual(len(node.input_tensors), 1)
    self.assertEqual(len(node.output_tensors), 1)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "reshape/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqual(node.output_tensors[0].shape.dims, [2, 5])
    self.assertEqual(node.output_tensors[0].shape.layout, NC)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

  def test_repeat_op(self):
    node = self.get_node(self.test_graph.graph, "repeat")
    self.assertEqual(node.op, Repeat)
    self.assertEqual(len(node.input_tensors), 1)
    self.assertEqual(len(node.output_tensors), 1)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "repeat/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqual(node.output_tensors[0].shape.dims, [8, 10])
    self.assertEqual(node.output_tensors[0].shape.layout, NC)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

  def test_stack_op(self):
    # stack op is implemented using expand_dims and repeat. Here we only test
    # the output.
    node = self.get_node(self.test_graph.graph, "stack:repeat")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqual(node.output_tensors[0].shape.dims, [8, 4, 10])
    self.assertEqual(node.output_tensors[0].shape.layout, NTC)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

  def test_unstack_op(self):
    # unstack op is implemented using split and squeeze. Here we only test
    # the output.
    for i in range(4):
      node = self.get_node(
          self.test_graph.graph,
          "unstack:squeeze" + ("_%d" % i if i > 0 else ""))
      self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
      self.assertEqual(node.output_tensors[0].shape.dims, [8, 10])
      self.assertEqual(node.output_tensors[0].shape.layout, NC)
      self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

class ResidualGraphTest(OperatorTest):
  """Common tests for the residual graph."""

  def test_input_op(self):
    node = self.get_node(self.test_graph.graph, "input")
    self.assertEqual(node.op, Data)
    self.assertEqual(len(node.input_tensors), 1)
    self.assertEqual(len(node.output_tensors), 1)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "input/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqual(node.output_tensors[0].shape.dims, [1, 1, 28, 28])
    self.assertEqual(node.output_tensors[0].shape.layout, NCHW)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

  def test_convolution_op(self):
    expected_weight_layout = backend_layouts[
        self.backend][Convolution3d].input_layoutsets[1].layouts
    expected_output_layout = backend_layouts[
        self.backend][Convolution3d].output_layoutset.layouts
    # The first convolution operator "conv0"
    node = self.get_node(self.test_graph.graph, "conv0")
    self.assertEqual(node.op, Convolution3d)
    self.assertEqual(len(node.input_tensors), 2)
    self.assertEqual(len(node.output_tensors), 1)
    # Parameters
    self.assertEqual(node.params.conv_params.padding, SamePadding)
    self.assertEqual(node.params.conv_params.stride, [1, 1])
    # Weight tensor
    self.assertEqual(node.input_tensors[1].data_type, self.expected_dtype)
    self.assertEqualDims(node.input_tensors[1].shape.dims,
                         node.input_tensors[1].shape.layout, [64, 1, 3, 3],
                         NCHW)
    self.assertEqual(node.input_tensors[1].shape.layout, expected_weight_layout)
    self.assertEqual(node.input_tensors[1].shape.alignment, self.alignment)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "conv0/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqualDims(node.output_tensors[0].shape.dims,
                         node.output_tensors[0].shape.layout, [1, 64, 28, 28],
                         NCHW)
    self.assertEqual(node.output_tensors[0].shape.layout,
                     expected_output_layout)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

    # The second convolution operator "conv1"
    node = self.get_node(self.test_graph.graph, "conv1")
    self.assertEqual(node.op, Convolution3d)
    self.assertEqual(len(node.input_tensors), 2)
    self.assertEqual(len(node.output_tensors), 1)
    # Parameters
    self.assertEqual(node.params.conv_params.padding, SamePadding)
    self.assertEqual(node.params.conv_params.stride, [1, 1])
    # Weight tensor
    self.assertEqual(node.input_tensors[1].data_type, self.expected_dtype)
    self.assertEqualDims(node.input_tensors[1].shape.dims,
                         node.input_tensors[1].shape.layout, [64, 1, 3, 3],
                         NCHW)
    self.assertEqual(node.input_tensors[1].shape.layout, expected_weight_layout)
    self.assertEqual(node.input_tensors[1].shape.alignment, self.alignment)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "conv1/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqualDims(node.output_tensors[0].shape.dims,
                         node.output_tensors[0].shape.layout, [1, 64, 28, 28],
                         NCHW)
    self.assertEqual(node.output_tensors[0].shape.layout,
                     expected_output_layout)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

    # The third convolution operator "conv2"
    node = self.get_node(self.test_graph.graph, "conv2")
    self.assertEqual(node.op, Convolution3d)
    self.assertEqual(len(node.input_tensors), 2)
    self.assertEqual(len(node.output_tensors), 1)
    # Parameters
    self.assertEqual(node.params.conv_params.padding, SamePadding)
    self.assertEqual(node.params.conv_params.stride, [1, 1])
    # Weight tensor
    self.assertEqual(node.input_tensors[1].data_type, self.expected_dtype)
    self.assertEqualDims(node.input_tensors[1].shape.dims,
                         node.input_tensors[1].shape.layout, [64, 64, 3, 3],
                         NCHW)
    self.assertEqual(node.input_tensors[1].shape.layout, expected_weight_layout)
    self.assertEqual(node.input_tensors[1].shape.alignment, self.alignment)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "conv2/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqualDims(node.output_tensors[0].shape.dims,
                         node.output_tensors[0].shape.layout, [1, 64, 28, 28],
                         NCHW)
    self.assertEqual(node.output_tensors[0].shape.layout,
                     expected_output_layout)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

  def test_relu_op(self):
    node = self.get_node(self.test_graph.graph, "relu")
    self.assertEqual(node.op, ReLU)
    self.assertEqual(len(node.input_tensors), 1)
    self.assertEqual(len(node.output_tensors), 1)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "relu/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqualDims(node.output_tensors[0].shape.dims,
                         node.output_tensors[0].shape.layout, [1, 64, 28, 28],
                         NCHW)
    self.assertEqual(node.output_tensors[0].shape.layout,
                     node.input_tensors[0].shape.layout)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

  def test_batch_norm_op(self):
    node = self.get_node(self.test_graph.graph, "bn")
    self.assertEqual(node.op, BatchNorm)
    self.assertEqual(len(node.input_tensors), 5)
    self.assertEqual(len(node.output_tensors), 1)
    # Weight tensors
    for i in [1, 2, 3, 4]:
      self.assertEqual(node.input_tensors[i].data_type, self.expected_dtype)
      self.assertEqual(node.input_tensors[i].shape.dims, [1, 64])
      self.assertEqual(node.input_tensors[i].shape.layout, NC)
      self.assertEqual(node.input_tensors[i].shape.alignment, self.alignment)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "bn/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqualDims(node.output_tensors[0].shape.dims,
                         node.output_tensors[0].shape.layout, [1, 64, 28, 28],
                         NCHW)
    self.assertEqual(node.output_tensors[0].shape.layout,
                     node.input_tensors[0].shape.layout)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

  def test_add_op(self):
    # The first add operator (add)
    node = self.get_node(self.test_graph.graph, "add")
    self.assertEqual(node.op, EltwiseAdd)
    self.assertEqual(len(node.input_tensors), 2)
    self.assertEqual(len(node.output_tensors), 1)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "add/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqualDims(node.output_tensors[0].shape.dims,
                         node.output_tensors[0].shape.layout, [1, 64, 28, 28],
                         NCHW)
    self.assertEqual(node.output_tensors[0].shape.layout,
                     node.input_tensors[0].shape.layout)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

    # The second add operator (add1)
    node = self.get_node(self.test_graph.graph, "add1")
    self.assertEqual(node.op, EltwiseAdd)
    self.assertEqual(len(node.input_tensors), 2)
    self.assertEqual(len(node.output_tensors), 1)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "add1/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqualDims(node.output_tensors[0].shape.dims,
                         node.output_tensors[0].shape.layout, [1, 32, 28, 28],
                         NCHW)
    self.assertEqual(node.output_tensors[0].shape.layout,
                     node.input_tensors[0].shape.layout)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

    # The third add operator (add2)
    node = self.get_node(self.test_graph.graph, "add2")
    self.assertEqual(node.op, EltwiseAdd)
    self.assertEqual(len(node.input_tensors), 2)
    self.assertEqual(len(node.output_tensors), 1)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "add2/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqualDims(node.output_tensors[0].shape.dims,
                         node.output_tensors[0].shape.layout, [1, 32, 28, 28],
                         NCHW)
    self.assertEqual(node.output_tensors[0].shape.layout,
                     node.input_tensors[0].shape.layout)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

  def test_mul_op(self):
    # The first mul operator (mul)
    node = self.get_node(self.test_graph.graph, "mul")
    self.assertEqual(node.op, EltwiseMul)
    self.assertEqual(len(node.input_tensors), 2)
    self.assertEqual(len(node.output_tensors), 1)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "mul/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqualDims(node.output_tensors[0].shape.dims,
                         node.output_tensors[0].shape.layout, [1, 64, 28, 28],
                         NCHW)
    self.assertEqual(node.output_tensors[0].shape.layout,
                     node.input_tensors[0].shape.layout)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

    # The second add operator (mul1)
    node = self.get_node(self.test_graph.graph, "mul1")
    self.assertEqual(node.op, EltwiseMul)
    self.assertEqual(len(node.input_tensors), 2)
    self.assertEqual(len(node.output_tensors), 1)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "mul1/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqualDims(node.output_tensors[0].shape.dims,
                         node.output_tensors[0].shape.layout, [1, 32, 28, 28],
                         NCHW)
    self.assertEqual(node.output_tensors[0].shape.layout,
                     node.input_tensors[0].shape.layout)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

  def test_concat_op(self):
    node = self.get_node(self.test_graph.graph, "concat")
    self.assertEqual(node.op, Concat)
    self.assertEqual(len(node.input_tensors), 2)
    self.assertEqual(len(node.output_tensors), 1)
    # Output tensor
    self.assertEqual(node.output_tensors[0].name, "concat/output0")
    self.assertEqual(node.output_tensors[0].data_type, self.expected_dtype)
    self.assertEqualDims(node.output_tensors[0].shape.dims,
                         node.output_tensors[0].shape.layout, [1, 128, 28, 28],
                         NCHW)
    self.assertEqual(node.output_tensors[0].shape.layout,
                     node.input_tensors[0].shape.layout)
    self.assertEqual(node.output_tensors[0].shape.alignment, self.alignment)

  def test_split_op(self):
    node = self.get_node(self.test_graph.graph, "split")
    self.assertEqual(node.op, Split)
    self.assertEqual(len(node.input_tensors), 1)
    self.assertEqual(len(node.output_tensors), 4)
    # Output tensors
    for i in range(4):
      self.assertEqual(node.output_tensors[i].name, "split/output%d" % i)
      self.assertEqual(node.output_tensors[i].data_type, self.expected_dtype)
      self.assertEqualDims(node.output_tensors[i].shape.dims,
                           node.output_tensors[i].shape.layout, [1, 32, 28, 28],
                           NCHW)
      self.assertEqual(node.output_tensors[i].shape.layout,
                       node.input_tensors[0].shape.layout)
      self.assertEqual(node.output_tensors[i].shape.alignment, self.alignment)

class SMVSequentialGraphTest(smaug_test.SmaugTest, SequentialGraphTest):
  """Test the sequential graph on the SMV backend."""

  def __init__(self, *args, **kwargs):
    super(SMVSequentialGraphTest, self).__init__(*args, **kwargs)
    self.build_test_sequential_graph("SMV")

  def test_parent_children(self):
    """Test the parent/child relationship in the graph.

    Because different backends may require different layout transformations
    between layers, so we delete this test from the above common tests.
    """
    # input (Data).
    node = self.get_node(self.test_graph.graph, "input")
    self.assertEqual(len(node.parents), 0)
    self.assertEqual(node.children[0], "reorder")
    # Reorder input from NCHW to NHWC.
    node = self.get_node(self.test_graph.graph, "reorder")
    self.assertEqual(node.parents[0], "input")
    self.assertEqual(node.children[0], "conv0")
    # conv0 (Convolution).
    node = self.get_node(self.test_graph.graph, "conv0")
    self.assertEqual(node.parents[0], "reorder")
    self.assertEqual(node.children[0], "conv0_relu")
    # conv0_relu (ReLU).
    node = self.get_node(self.test_graph.graph, "conv0_relu")
    self.assertEqual(node.parents[0], "conv0")
    self.assertEqual(node.children[0], "bn")
    # bn (BN).
    node = self.get_node(self.test_graph.graph, "bn")
    self.assertEqual(node.parents[0], "conv0_relu")
    self.assertEqual(node.children[0], "conv1")
    # conv1 (Convolution).
    node = self.get_node(self.test_graph.graph, "conv1")
    self.assertEqual(node.parents[0], "bn")
    self.assertEqual(node.children[0], "conv1_relu")
    # conv1_relu (ReLU).
    node = self.get_node(self.test_graph.graph, "conv1_relu")
    self.assertEqual(node.parents[0], "conv1")
    self.assertEqual(node.children[0], "pool")
    # pool (MaxPooling).
    node = self.get_node(self.test_graph.graph, "pool")
    self.assertEqual(node.parents[0], "conv1_relu")
    self.assertEqual(node.children[0], "flatten")
    # flatten (Flatten).
    node = self.get_node(self.test_graph.graph, "flatten")
    self.assertEqual(node.parents[0], "pool")
    self.assertEqual(node.children[0], "fc0")
    # fc0 (FC).
    node = self.get_node(self.test_graph.graph, "fc0")
    self.assertEqual(node.parents[0], "flatten")
    self.assertEqual(node.children[0], "fc0_relu")
    # fc0_relu (ReLU)
    node = self.get_node(self.test_graph.graph, "fc0_relu")
    self.assertEqual(node.parents[0], "fc0")
    self.assertEqual(node.children[0], "fc1")
    # fc1 (FC).
    node = self.get_node(self.test_graph.graph, "fc1")
    self.assertEqual(node.parents[0], "fc0_relu")
    self.assertEqual(node.children[0], "expand_dims")
    # expand_dims (Reshape).
    node = self.get_node(self.test_graph.graph, "expand_dims")
    self.assertEqual(node.parents[0], "fc1")
    self.assertEqual(node.children[0], "squeeze")
    # squeeze (Reshape).
    node = self.get_node(self.test_graph.graph, "squeeze")
    self.assertEqual(node.parents[0], "expand_dims")
    self.assertEqual(node.children[0], "reshape")
    # reshape (Reshape).
    node = self.get_node(self.test_graph.graph, "reshape")
    self.assertEqual(node.parents[0], "squeeze")
    self.assertEqual(node.children[0], "repeat")
    # repeat (Repeat).
    node = self.get_node(self.test_graph.graph, "repeat")
    self.assertEqual(node.parents[0], "reshape")
    self.assertEqual(node.children[0], "stack:expand_dims")
    # stack (Reshape + Repeat).
    node = self.get_node(self.test_graph.graph, "stack:expand_dims")
    self.assertEqual(node.parents[0], "repeat")
    self.assertEqual(node.children[0], "stack:repeat")
    node = self.get_node(self.test_graph.graph, "stack:repeat")
    self.assertEqual(node.parents[0], "stack:expand_dims")
    self.assertEqual(node.children[0], "unstack:split")
    # unstack (Split + Squeeze).
    node = self.get_node(self.test_graph.graph, "unstack:split")
    self.assertEqual(node.parents[0], "stack:repeat")
    self.assertEqual(len(node.children), 4)
    for i, child in enumerate(node.children):
      child_name = "unstack:squeeze" + ("_%d" % i if i > 0 else "")
      self.assertEqual(child, child_name)
      child_node = self.get_node(self.test_graph.graph, child_name)
      self.assertEqual(child_node.parents[0], "unstack:split")
      self.assertEqual(child_node.src_tensors_indices, [i])
      self.assertEqual(len(child_node.children), 0)

class RefSequentialGraphTest(smaug_test.SmaugTest, SequentialGraphTest):
  """Test the sequential graph on the reference backend.

  This test should have no reorder operators because all the inputs are
  already in NCHW format.
  """

  def __init__(self, *args, **kwargs):
    super(RefSequentialGraphTest, self).__init__(*args, **kwargs)
    self.build_test_sequential_graph("Reference")

  def test_parent_children(self):
    """Test the parent/child relationship in the graph."""

    # input (Data).
    node = self.get_node(self.test_graph.graph, "input")
    self.assertEqual(len(node.parents), 0)
    self.assertEqual(node.children[0], "conv0")
    # conv0 (Convolution).
    node = self.get_node(self.test_graph.graph, "conv0")
    self.assertEqual(node.parents[0], "input")
    self.assertEqual(node.children[0], "conv0_relu")
    # conv0_relu (ReLU).
    node = self.get_node(self.test_graph.graph, "conv0_relu")
    self.assertEqual(node.parents[0], "conv0")
    self.assertEqual(node.children[0], "bn")
    # bn (BN)
    node = self.get_node(self.test_graph.graph, "bn")
    self.assertEqual(node.parents[0], "conv0_relu")
    self.assertEqual(node.children[0], "conv1")
    # conv1 (Convolution).
    node = self.get_node(self.test_graph.graph, "conv1")
    self.assertEqual(node.parents[0], "bn")
    self.assertEqual(node.children[0], "conv1_relu")
    # conv1_relu (ReLU)
    node = self.get_node(self.test_graph.graph, "conv1_relu")
    self.assertEqual(node.parents[0], "conv1")
    self.assertEqual(node.children[0], "pool")
    # pool (MaxPooling)
    node = self.get_node(self.test_graph.graph, "pool")
    self.assertEqual(node.parents[0], "conv1_relu")
    self.assertEqual(node.children[0], "flatten")
    # flatten (Flatten)
    node = self.get_node(self.test_graph.graph, "flatten")
    self.assertEqual(node.parents[0], "pool")
    self.assertEqual(node.children[0], "fc0")
    # Transpose fc0 weights
    node = self.get_node(self.test_graph.graph, "reorder")
    self.assertEqual(node.children[0], "fc0")
    # fc0 (FC)
    node = self.get_node(self.test_graph.graph, "fc0")
    self.assertEqual(node.parents, ["flatten", "reorder"])
    self.assertEqual(node.children[0], "fc0_relu")
    # fc0_relu (ReLU)
    node = self.get_node(self.test_graph.graph, "fc0_relu")
    self.assertEqual(node.parents[0], "fc0")
    self.assertEqual(node.children[0], "fc1")
    # Transpose fc1/weights
    node = self.get_node(self.test_graph.graph, "reorder_1")
    self.assertEqual(node.children[0], "fc1")
    # fc1 (FC)
    node = self.get_node(self.test_graph.graph, "fc1")
    self.assertEqual(node.parents, ["fc0_relu", "reorder_1"])
    self.assertEqual(node.children[0], "expand_dims")
    # expand_dims (Reshape).
    node = self.get_node(self.test_graph.graph, "expand_dims")
    self.assertEqual(node.parents[0], "fc1")
    self.assertEqual(node.children[0], "squeeze")
    # squeeze (Reshape).
    node = self.get_node(self.test_graph.graph, "squeeze")
    self.assertEqual(node.parents[0], "expand_dims")
    self.assertEqual(node.children[0], "reshape")
    # reshape (Reshape)
    node = self.get_node(self.test_graph.graph, "reshape")
    self.assertEqual(node.parents[0], "squeeze")
    self.assertEqual(node.children[0], "repeat")
    # repeat (Repeat)
    node = self.get_node(self.test_graph.graph, "repeat")
    self.assertEqual(node.parents[0], "reshape")
    self.assertEqual(node.children[0], "stack:expand_dims")
    # stack (Reshape + Repeat).
    node = self.get_node(self.test_graph.graph, "stack:expand_dims")
    self.assertEqual(node.parents[0], "repeat")
    self.assertEqual(node.children[0], "stack:repeat")
    node = self.get_node(self.test_graph.graph, "stack:repeat")
    self.assertEqual(node.parents[0], "stack:expand_dims")
    self.assertEqual(node.children[0], "unstack:split")
    # unstack (Split + Squeeze).
    node = self.get_node(self.test_graph.graph, "unstack:split")
    self.assertEqual(node.parents[0], "stack:repeat")
    self.assertEqual(len(node.children), 4)
    for i, child in enumerate(node.children):
      child_name = "unstack:squeeze" + ("_%d" % i if i > 0 else "")
      self.assertEqual(child, child_name)
      child_node = self.get_node(self.test_graph.graph, child_name)
      self.assertEqual(child_node.parents[0], "unstack:split")
      self.assertEqual(child_node.src_tensors_indices, [i])
      self.assertEqual(len(child_node.children), 0)

class SMVResidualGraphTest(smaug_test.SmaugTest, ResidualGraphTest):
  """Test the residual graph on the SMV backend."""

  def __init__(self, *args, **kwargs):
    super(SMVResidualGraphTest, self).__init__(*args, **kwargs)
    self.build_test_residual_graph("SMV")

  def test_parent_children(self):
    """Test the parent/child relationship in the graph."""

    # input (Data).
    node = self.get_node(self.test_graph.graph, "input")
    self.assertEqual(len(node.parents), 0)
    self.assertEqual(node.children[0], "reorder")
    # Reorder input from NCHW to NHWC.
    node = self.get_node(self.test_graph.graph, "reorder")
    self.assertEqual(node.parents[0], "input")
    self.assertEqual(node.children, ["conv0", "conv1"])
    # conv0 (Convolution).
    node = self.get_node(self.test_graph.graph, "conv0")
    self.assertEqual(node.parents[0], "reorder")
    self.assertEqual(node.children[0], "add")
    # conv1 (Convolution).
    node = self.get_node(self.test_graph.graph, "conv1")
    self.assertEqual(node.parents[0], "reorder")
    self.assertEqual(node.children[0], "bn")
    # bn (BN).
    node = self.get_node(self.test_graph.graph, "bn")
    self.assertEqual(node.parents[0], "conv1")
    self.assertEqual(node.children[0], "relu")
    # relu (ReLU).
    node = self.get_node(self.test_graph.graph, "relu")
    self.assertEqual(node.parents[0], "bn")
    self.assertEqual(node.children[0], "conv2")
    # conv2 (Convolution).
    node = self.get_node(self.test_graph.graph, "conv2")
    self.assertEqual(node.parents[0], "relu")
    self.assertEqual(node.children[0], "add")
    # add (EltwiseAdd).
    node = self.get_node(self.test_graph.graph, "add")
    self.assertEqual(node.parents[0], "conv0")
    self.assertEqual(node.parents[1], "conv2")
    self.assertEqual(node.children[0], "mul")
    # mul (EltwiseMul).
    node = self.get_node(self.test_graph.graph, "mul")
    self.assertEqual(node.parents, ["conv0", "add"])
    self.assertEqual(node.children[0], "concat")
    # concat (Concat).
    node = self.get_node(self.test_graph.graph, "concat")
    self.assertEqual(node.parents, ["conv0", "mul"])
    self.assertEqual(node.children[0], "split")
    # split (Split).
    node = self.get_node(self.test_graph.graph, "split")
    self.assertEqual(node.parents[0], "concat")
    self.assertEqual(node.children, ["add1", "add2"])
    # add1 (EltwiseAdd).
    node = self.get_node(self.test_graph.graph, "add1")
    self.assertEqual(node.parents, ["split", "split"])
    self.assertEqual(node.src_tensors_indices, [0, 1])
    self.assertEqual(node.children[0], "mul1")
    # add2 (EltwiseAdd).
    node = self.get_node(self.test_graph.graph, "add2")
    self.assertEqual(node.parents, ["split", "split"])
    self.assertEqual(node.src_tensors_indices, [2, 3])
    self.assertEqual(node.children[0], "mul1")
    # mul (EltwiseMul).
    node = self.get_node(self.test_graph.graph, "mul1")
    self.assertEqual(node.parents, ["add1", "add2"])
    self.assertEqual(len(node.children), 0)

class RefResidualGraphTest(smaug_test.SmaugTest, ResidualGraphTest):
  """Test the residual graph on the reference backend."""

  def __init__(self, *args, **kwargs):
    super(RefResidualGraphTest, self).__init__(*args, **kwargs)
    self.build_test_residual_graph("Reference")

  def test_parent_children(self):
    """Test the parent/child relationship in the graph."""

    # input (Data).
    node = self.get_node(self.test_graph.graph, "input")
    self.assertEqual(len(node.parents), 0)
    self.assertEqual(node.children, ["conv0", "conv1"])
    # conv0 (Convolution).
    node = self.get_node(self.test_graph.graph, "conv0")
    self.assertEqual(node.parents[0], "input")
    self.assertEqual(node.children[0], "add")
    # conv1 (Convolution).
    node = self.get_node(self.test_graph.graph, "conv1")
    self.assertEqual(node.parents[0], "input")
    self.assertEqual(node.children[0], "bn")
    # bn (BN).
    node = self.get_node(self.test_graph.graph, "bn")
    self.assertEqual(node.parents[0], "conv1")
    self.assertEqual(node.children[0], "relu")
    # relu (ReLU).
    node = self.get_node(self.test_graph.graph, "relu")
    self.assertEqual(node.parents[0], "bn")
    self.assertEqual(node.children[0], "conv2")
    # conv2 (Convolution).
    node = self.get_node(self.test_graph.graph, "conv2")
    self.assertEqual(node.parents[0], "relu")
    self.assertEqual(node.children[0], "add")
    # add (EltwiseAdd).
    node = self.get_node(self.test_graph.graph, "add")
    self.assertEqual(node.parents, ["conv0", "conv2"])
    self.assertEqual(node.children[0], "mul")
    # mul (EltwiseMul).
    node = self.get_node(self.test_graph.graph, "mul")
    self.assertEqual(node.parents, ["conv0", "add"])
    self.assertEqual(node.children[0], "concat")
    # concat (Concat).
    node = self.get_node(self.test_graph.graph, "concat")
    self.assertEqual(node.parents, ["conv0", "mul"])
    self.assertEqual(node.children[0], "split")
    # split (Split).
    node = self.get_node(self.test_graph.graph, "split")
    self.assertEqual(node.parents[0], "concat")
    self.assertEqual(node.children, ["add1", "add2"])
    # add1 (EltwiseAdd).
    node = self.get_node(self.test_graph.graph, "add1")
    self.assertEqual(node.parents, ["split", "split"])
    self.assertEqual(node.src_tensors_indices, [0, 1])
    self.assertEqual(node.children[0], "mul1")
    # add2 (EltwiseAdd).
    node = self.get_node(self.test_graph.graph, "add2")
    self.assertEqual(node.parents, ["split", "split"])
    self.assertEqual(node.src_tensors_indices, [2, 3])
    self.assertEqual(node.children[0], "mul1")
    # mul (EltwiseMul).
    node = self.get_node(self.test_graph.graph, "mul1")
    self.assertEqual(node.parents, ["add1", "add2"])
    self.assertEqual(len(node.children), 0)

if __name__ == "__main__":
  unittest.main()
