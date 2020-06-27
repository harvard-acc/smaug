#!/usr/bin/env python

"""Tests for activation functions."""

import unittest
import numpy as np

from smaug.python.graph import Graph
from smaug.python.tensor import Tensor
from smaug.python.ops import data_op
from smaug.python.ops import activation_ops
from smaug.python.ops import array_ops
from smaug.python.ops import nn_ops
from smaug.core import types_pb2

class ActivationFunctionTest(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super(ActivationFunctionTest, self).__init__(*args, **kwargs)
    self.tensor_shape = [2, 32, 32, 32]

  def do_basic_test(
      self, test_graph, node_name, op_type, child_name=None, tensor_shape=None):
    node = test_graph.get_node(node_name)
    if tensor_shape == None:
      tensor_shape = self.tensor_shape
    self.assertEqual(node.op, op_type)
    self.assertEqual(len(node.input_tensors), 1)
    self.assertEqual(len(node.output_tensors), 1)
    self.assertEqual(node.output_tensors[0].data_type, types_pb2.Float16)
    self.assertEqual(node.output_tensors[0].shape.dims, tensor_shape)
    self.assertEqual(node.output_tensors[0].shape.layout,
                     node.input_tensors[0].shape.layout)
    self.assertEqual(node.output_tensors[0].shape.alignment, 8)
    if len(node.children) > 0:
      self.assertEqual(node.children[0], child_name)
    return node

  def test_activation_functions(self):
    """Test activation function attributes."""
    with Graph("test_graph", "SMV") as test_graph:
      tensor_data = np.random.rand(*self.tensor_shape).astype(np.float16)
      input_tensor = Tensor(data_layout=types_pb2.NHWC, tensor_data=tensor_data)
      act = data_op.input_data(input_tensor, "input")
      act = activation_ops.relu(act, "relu")
      act = activation_ops.lrelu(act, slope=0.5, name="lrelu")
      act = activation_ops.elu(act, alpha=0.2, name="elu")
      act = activation_ops.selu(act, alpha=0.4, lambda_param=0.8, name="selu")
      act = activation_ops.tanh(act, "tanh")
      act = activation_ops.hard_tanh(act, min=-1.5, max=1.5, name="hard_tanh")
      act = activation_ops.sigmoid(act, "sigmoid")
      # Softmax expects NC format, so reorder NHWC to NC.
      act = array_ops.reorder(act, target_layout=types_pb2.NC, name="reorder")
      act = activation_ops.softmax(act, "softmax")
    # ReLU
    self.do_basic_test(test_graph, "relu", types_pb2.ReLU, "lrelu")
    # LReLU
    node = self.do_basic_test(test_graph, "lrelu", types_pb2.LReLU, "elu")
    self.assertAlmostEqual(node.params.act_params.lrelu_params.slope, 0.5)
    # ELU
    node = self.do_basic_test(test_graph, "elu", types_pb2.ELU, "selu")
    self.assertAlmostEqual(node.params.act_params.elu_params.alpha, 0.2)
    # SELU
    node = self.do_basic_test(test_graph, "selu", types_pb2.SELU, "tanh")
    self.assertAlmostEqual(node.params.act_params.elu_params.alpha, 0.4)
    self.assertAlmostEqual(node.params.act_params.elu_params.lambda_param, 0.8)
    # Tanh
    self.do_basic_test(test_graph, "tanh", types_pb2.Tanh, "hard_tanh")
    # HardTanh
    node = self.do_basic_test(
        test_graph, "hard_tanh", types_pb2.HardTanh, "sigmoid")
    self.assertAlmostEqual(node.params.act_params.hard_tanh_params.min, -1.5)
    self.assertAlmostEqual(node.params.act_params.hard_tanh_params.max, 1.5)
    # Sigmoid
    self.do_basic_test(test_graph, "sigmoid", types_pb2.Sigmoid, "reorder")
    # Softmax
    self.do_basic_test(
        test_graph, "softmax", types_pb2.Softmax, tensor_shape=[2, 32768])

  def test_fusing_activation_functions(self):
    """Test activation function when they are fused with other operators."""
    with Graph("test_graph", "SMV") as test_graph:
      input_tensor = Tensor(
          data_layout=types_pb2.NHWC,
          tensor_data=np.random.rand(*self.tensor_shape).astype(np.float16))
      filter_tensor = Tensor(
          data_layout=types_pb2.NHWC,
          tensor_data=np.random.rand(32, 3, 3, 32).astype(np.float16))
      weight_tensor = Tensor(
          data_layout=types_pb2.NC,
          tensor_data=np.random.rand(10, 32768).astype(np.float16))
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
      act = data_op.input_data(input_tensor, "input")
      act = nn_ops.convolution(
          act, filter_tensor, stride=[1, 1], padding="same", activation=None,
          name="conv_none")
      act = nn_ops.convolution(
          act, filter_tensor, stride=[1, 1], padding="same",
          activation=types_pb2.ReLU, name="conv_relu")
      act = nn_ops.convolution(
          act, filter_tensor, stride=[1, 1], padding="same",
          activation=types_pb2.LReLU, name="conv_lrelu")
      act = nn_ops.convolution(
          act, filter_tensor, stride=[1, 1], padding="same",
          activation=types_pb2.ELU, name="conv_elu")
      act = nn_ops.convolution(
          act, filter_tensor, stride=[1, 1], padding="same",
          activation=types_pb2.SELU, name="conv_selu")
      act = nn_ops.convolution(
          act, filter_tensor, stride=[1, 1], padding="same",
          activation=types_pb2.Tanh, name="conv_tanh")
      act = nn_ops.convolution(
          act, filter_tensor, stride=[1, 1], padding="same",
          activation=types_pb2.HardTanh, name="conv_hard_tanh")
      act = nn_ops.convolution(
          act, filter_tensor, stride=[1, 1], padding="same",
          activation=types_pb2.Sigmoid, name="conv_sigmoid")
      act = nn_ops.convolution(
          act, filter_tensor, stride=[1, 1], padding="same",
          activation=types_pb2.Softmax, name="conv_softmax")
      act = nn_ops.batch_norm(
          act, bn_mean_tensor, bn_var_tensor, bn_gamma_tensor, bn_beta_tensor,
          activation=types_pb2.ReLU, name="bn_relu")
      act = nn_ops.mat_mul(
          act, weight_tensor, activation=types_pb2.ReLU, name="fc_relu")
    # None
    node = test_graph.get_node("conv_none")
    self.assertEqual(node.params.act_params.activation, types_pb2.UnknownOp)
    # ReLU
    node = test_graph.get_node("conv_relu")
    self.assertEqual(node.params.act_params.activation, types_pb2.ReLU)
    # LReLU (default slope = 0.2)
    node = test_graph.get_node("conv_lrelu")
    self.assertEqual(node.params.act_params.activation, types_pb2.LReLU)
    self.assertAlmostEqual(node.params.act_params.lrelu_params.slope, 0.2)
    # ELU (default alpha = 0.1)
    node = test_graph.get_node("conv_elu")
    self.assertEqual(node.params.act_params.activation, types_pb2.ELU)
    self.assertAlmostEqual(node.params.act_params.elu_params.alpha, 0.1)
    # SELU (default alpha = 1.6733, lambda = 1.0507)
    node = test_graph.get_node("conv_selu")
    self.assertEqual(node.params.act_params.activation, types_pb2.SELU)
    self.assertAlmostEqual(node.params.act_params.elu_params.alpha, 1.6733, 5)
    self.assertAlmostEqual(node.params.act_params.elu_params.lambda_param,
                           1.0507, 5)
    # Tanh
    node = test_graph.get_node("conv_tanh")
    self.assertEqual(node.params.act_params.activation, types_pb2.Tanh)
    # HardTanh (default min = -1, max = 1)
    node = test_graph.get_node("conv_hard_tanh")
    self.assertEqual(node.params.act_params.activation, types_pb2.HardTanh)
    self.assertAlmostEqual(node.params.act_params.hard_tanh_params.min, -1)
    self.assertAlmostEqual(node.params.act_params.hard_tanh_params.max, 1)
    # Sigmoid
    node = test_graph.get_node("conv_sigmoid")
    self.assertEqual(node.params.act_params.activation, types_pb2.Sigmoid)
    # Softmax
    node = test_graph.get_node("conv_softmax")
    self.assertEqual(node.params.act_params.activation, types_pb2.Softmax)
    # Fusion with inner products and batch norms.
    node = test_graph.get_node("bn_relu")
    self.assertEqual(node.params.act_params.activation, types_pb2.ReLU)
    node = test_graph.get_node("fc_relu")
    self.assertEqual(node.params.act_params.activation, types_pb2.ReLU)

if __name__ == "__main__":
  unittest.main()
