#!/usr/bin/env python

""" This tests operators that can potentially lead to FP precision loss."""

import unittest
import tensorflow as tf
import numpy as np

from smaug.core import types_pb2
from smaug.python.smaug_test import SmaugTest
from smaug.python.graph import Graph
from smaug.python.tensor import Tensor
from smaug.python.ops import nn_ops
from smaug.python.ops import math_ops

initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)

class FpPrecisionTest(SmaugTest):
  def test_mat_mul(self):
    batch = 4
    channels = 32
    units = 128
    tf_a = tf.Variable(initializer(shape=[batch, channels], dtype=self.dtype))
    tf_b = tf.Variable(initializer(shape=[units, channels], dtype=self.dtype))
    tf_result = tf.linalg.matmul(tf_a, tf_b, transpose_b=True)

    a = Tensor(data_layout=types_pb2.NC, tensor_data=tf_a.numpy())
    b = Tensor(data_layout=types_pb2.NC, tensor_data=tf_b.numpy())
    with Graph(name=self.graph_name, backend=self.backend) as graph:
      nn_ops.mat_mul(a, b)
    self.runAndValidate(graph, tf_result, decimal=3)

  def test_convolution(self):
    batch = 4
    width = 8
    height = 8
    channels = 32
    filter_height = 3
    filter_width = 3
    num_filters = 128
    tf_inputs = tf.Variable(
        initializer(shape=[batch, height, width, channels], dtype=self.dtype))
    tf_filters = tf.Variable(
        initializer(
            shape=[filter_height, filter_width, channels, num_filters],
            dtype=self.dtype))
    tf_results = tf.nn.conv2d(
        tf_inputs, tf_filters, strides=[1, 1], padding="SAME",
        data_format="NHWC", dilations=None)

    inputs = Tensor(data_layout=types_pb2.NHWC, tensor_data=tf_inputs.numpy())
    filters = Tensor(
        data_layout=types_pb2.NHWC,
        tensor_data=np.transpose(tf_filters.numpy(), (3, 0, 1, 2)))
    with Graph(name=self.graph_name, backend=self.backend) as graph:
      nn_ops.convolution(inputs, filters, stride=[1, 1], padding="same")
    self.runAndValidate(graph, tf_results, decimal=2)

  def test_add(self):
    batch = 4
    channels = 32
    tf_a = tf.Variable(initializer(shape=[batch, channels], dtype=self.dtype))
    tf_b = tf.Variable(initializer(shape=[batch, channels], dtype=self.dtype))
    tf_result = tf.math.add(tf_a, tf_b)

    a = Tensor(data_layout=types_pb2.NC, tensor_data=tf_a.numpy())
    b = Tensor(data_layout=types_pb2.NC, tensor_data=tf_b.numpy())
    with Graph(name=self.graph_name, backend=self.backend) as graph:
      math_ops.add(a, b)
    self.runAndValidate(graph, tf_result, decimal=3)

  def test_mul(self):
    batch = 4
    channels = 32
    tf_a = tf.Variable(initializer(shape=[batch, channels], dtype=self.dtype))
    tf_b = tf.Variable(initializer(shape=[batch, channels], dtype=self.dtype))
    tf_result = tf.math.multiply(tf_a, tf_b)

    a = Tensor(data_layout=types_pb2.NC, tensor_data=tf_a.numpy())
    b = Tensor(data_layout=types_pb2.NC, tensor_data=tf_b.numpy())
    with Graph(name=self.graph_name, backend=self.backend) as graph:
      math_ops.mul(a, b)
    self.runAndValidate(graph, tf_result, decimal=3)

if __name__ == "__main__":
  unittest.main()
