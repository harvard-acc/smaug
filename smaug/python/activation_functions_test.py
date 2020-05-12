#!/usr/bin/env python

"""Tests for activation functions."""

import unittest

from smaug.python import smaug_test
from smaug.python.graph import Graph
from smaug.python.tensor import Tensor
from smaug.python.ops import *
from smaug.core.types_pb2 import *

class ActivationFunctionTest(smaug_test.SmaugTest):
  def __init__(self, *args, **kwargs):
    super(ActivationFunctionTest, self).__init__(*args, **kwargs)
    self.tensor_shape = [2, 32, 32, 32]

  def do_basic_test(
      self, test_graph, node_name, op_type, child_name=None, tensor_shape=None):
    node = self.get_node(test_graph.graph, node_name)
    if tensor_shape == None:
      tensor_shape = self.tensor_shape
    self.assertEqual(node.op, op_type)
    self.assertEqual(len(node.input_tensors), 1)
    self.assertEqual(len(node.output_tensors), 1)
    self.assertEqual(node.output_tensors[0].data_type, Float16)
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
      input_tensor = Tensor(data_layout=NHWC, tensor_data=tensor_data)
      act = input_data(input_tensor, "input")
      act = relu(act, "relu")
      act = lrelu(act, slope=0.5, name="lrelu")
      act = elu(act, alpha=0.2, name="elu")
      act = selu(act, alpha=0.4, lambda_param=0.8, name="selu")
      act = tanh(act, "tanh")
      act = hard_tanh(act, min=-1.5, max=1.5, name="hard_tanh")
      act = sigmoid(act, "sigmoid")
      # Softmax expects NC format, so reorder NHWC to NC.
      act = reorder(act, target_layout=NC, name="reorder")
      act = softmax(act, "softmax")
    # ReLU
    self.do_basic_test(test_graph, "relu", ReLU, "lrelu")
    # LReLU
    node = self.do_basic_test(test_graph, "lrelu", LReLU, "elu")
    self.assertAlmostEqual(node.params.act_params.lrelu_params.slope, 0.5)
    # ELU
    node = self.do_basic_test(test_graph, "elu", ELU, "selu")
    self.assertAlmostEqual(node.params.act_params.elu_params.alpha, 0.2)
    # SELU
    node = self.do_basic_test(test_graph, "selu", SELU, "tanh")
    self.assertAlmostEqual(node.params.act_params.elu_params.alpha, 0.4)
    self.assertAlmostEqual(node.params.act_params.elu_params.lambda_param, 0.8)
    # Tanh
    self.do_basic_test(test_graph, "tanh", Tanh, "hard_tanh")
    # HardTanh
    node = self.do_basic_test(test_graph, "hard_tanh", HardTanh, "sigmoid")
    self.assertAlmostEqual(node.params.act_params.hard_tanh_params.min, -1.5)
    self.assertAlmostEqual(node.params.act_params.hard_tanh_params.max, 1.5)
    # Sigmoid
    self.do_basic_test(test_graph, "sigmoid", Sigmoid, "reorder")
    # Softmax
    self.do_basic_test(test_graph, "softmax", Softmax, tensor_shape=[2, 32768])

  def test_fusing_activation_functions(self):
    """Test activation function when they are fused with other operators."""
    with Graph("test_graph", "SMV") as test_graph:
      input_tensor = Tensor(
          data_layout=NHWC,
          tensor_data=np.random.rand(*self.tensor_shape).astype(np.float16))
      filter_tensor = Tensor(
          data_layout=NHWC,
          tensor_data=np.random.rand(32, 3, 3, 32).astype(np.float16))
      weight_tensor = Tensor(
          data_layout=NC,
          tensor_data=np.random.rand(10, 32768).astype(np.float16))
      bn_mean_tensor = Tensor(
          data_layout=NC, tensor_data=np.random.rand(1, 64).astype(np.float16))
      bn_var_tensor = Tensor(
          data_layout=NC, tensor_data=np.random.rand(1, 64).astype(np.float16))
      bn_gamma_tensor = Tensor(
          data_layout=NC, tensor_data=np.random.rand(1, 64).astype(np.float16))
      bn_beta_tensor = Tensor(
          data_layout=NC, tensor_data=np.random.rand(1, 64).astype(np.float16))
      act = input_data(input_tensor, "input")
      act = convolution(
          act, filter_tensor, stride=[1, 1], padding="same", activation=None,
          name="conv_none")
      act = convolution(
          act, filter_tensor, stride=[1, 1], padding="same", activation=ReLU,
          name="conv_relu")
      act = convolution(
          act, filter_tensor, stride=[1, 1], padding="same", activation=LReLU,
          name="conv_lrelu")
      act = convolution(
          act, filter_tensor, stride=[1, 1], padding="same", activation=ELU,
          name="conv_elu")
      act = convolution(
          act, filter_tensor, stride=[1, 1], padding="same", activation=SELU,
          name="conv_selu")
      act = convolution(
          act, filter_tensor, stride=[1, 1], padding="same", activation=Tanh,
          name="conv_tanh")
      act = convolution(
          act, filter_tensor, stride=[1, 1], padding="same",
          activation=HardTanh, name="conv_hard_tanh")
      act = convolution(
          act, filter_tensor, stride=[1, 1], padding="same", activation=Sigmoid,
          name="conv_sigmoid")
      act = convolution(
          act, filter_tensor, stride=[1, 1], padding="same", activation=Softmax,
          name="conv_softmax")
      act = batch_norm(
          act, bn_mean_tensor, bn_var_tensor, bn_gamma_tensor, bn_beta_tensor,
          activation=ReLU, name="bn_relu")
      act = mat_mul(act, weight_tensor, activation=ReLU, name="fc_relu")
    # None
    node = self.get_node(test_graph.graph, "conv_none")
    self.assertEqual(node.params.act_params.activation, UnknownOp)
    # ReLU
    node = self.get_node(test_graph.graph, "conv_relu")
    self.assertEqual(node.params.act_params.activation, ReLU)
    # LReLU (default slope = 0.2)
    node = self.get_node(test_graph.graph, "conv_lrelu")
    self.assertEqual(node.params.act_params.activation, LReLU)
    self.assertAlmostEqual(node.params.act_params.lrelu_params.slope, 0.2)
    # ELU (default alpha = 0.1)
    node = self.get_node(test_graph.graph, "conv_elu")
    self.assertEqual(node.params.act_params.activation, ELU)
    self.assertAlmostEqual(node.params.act_params.elu_params.alpha, 0.1)
    # SELU (default alpha = 1.6733, lambda = 1.0507)
    node = self.get_node(test_graph.graph, "conv_selu")
    self.assertEqual(node.params.act_params.activation, SELU)
    self.assertAlmostEqual(node.params.act_params.elu_params.alpha, 1.6733, 5)
    self.assertAlmostEqual(node.params.act_params.elu_params.lambda_param,
                           1.0507, 5)
    # Tanh
    node = self.get_node(test_graph.graph, "conv_tanh")
    self.assertEqual(node.params.act_params.activation, Tanh)
    # HardTanh (default min = -1, max = 1)
    node = self.get_node(test_graph.graph, "conv_hard_tanh")
    self.assertEqual(node.params.act_params.activation, HardTanh)
    self.assertAlmostEqual(node.params.act_params.hard_tanh_params.min, -1)
    self.assertAlmostEqual(node.params.act_params.hard_tanh_params.max, 1)
    # Sigmoid
    node = self.get_node(test_graph.graph, "conv_sigmoid")
    self.assertEqual(node.params.act_params.activation, Sigmoid)
    # Softmax
    node = self.get_node(test_graph.graph, "conv_softmax")
    self.assertEqual(node.params.act_params.activation, Softmax)
    # Fusion with inner products and batch norms.
    node = self.get_node(test_graph.graph, "bn_relu")
    self.assertEqual(node.params.act_params.activation, ReLU)
    node = self.get_node(test_graph.graph, "fc_relu")
    self.assertEqual(node.params.act_params.activation, ReLU)

if __name__ == "__main__":
  unittest.main()
