#!/usr/bin/env python

import unittest
import numpy as np

from smaug.python.smaug_test import SmaugTest
from smaug.python import global_vars
from smaug.python.graph import Graph
from smaug.python.tensor import Tensor
from smaug.python.ops import math_ops
from smaug.python.ops import data_op
from smaug.python.ops import control_flow_ops
from smaug.core import types_pb2

class ControlFlowOpsTest(SmaugTest):
  def test_cond_op_simple_func(self):
    with Graph(name=self.graph_name, backend=self.backend) as graph:
      x0 = Tensor(
          data_layout=types_pb2.N, tensor_data=np.array([2], dtype=self.dtype))
      x1 = Tensor(
          data_layout=types_pb2.N, tensor_data=np.array([5], dtype=self.dtype))
      y = Tensor(
          data_layout=types_pb2.N, tensor_data=np.array([10], dtype=self.dtype))
      z = Tensor(
          data_layout=types_pb2.N, tensor_data=np.array([20], dtype=self.dtype))
      expected_res = Tensor(
          data_layout=types_pb2.N, tensor_data=np.array([30], dtype=self.dtype))
      # res = y + z if x0 < x1 else y * z
      res = control_flow_ops.cond(
          math_ops.less(x0, x1), lambda: math_ops.add(y, z),
          lambda: math_ops.mul(y, z))
    self.runAndValidate(graph, expected_res.tensor_data)

  def test_cond_op_func_call(self):
    def func(a, b):
      minus_three = Tensor(
          data_layout=types_pb2.N, tensor_data=np.array([-3], dtype=self.dtype))
      return math_ops.add(a, math_ops.mul(b, minus_three))

    with Graph(name=self.graph_name, backend=self.backend) as graph:
      x0 = Tensor(
          data_layout=types_pb2.N, tensor_data=np.array([2], dtype=self.dtype))
      x1 = Tensor(
          data_layout=types_pb2.N, tensor_data=np.array([5], dtype=self.dtype))
      y = Tensor(
          data_layout=types_pb2.N, tensor_data=np.array([10], dtype=self.dtype))
      z = Tensor(
          data_layout=types_pb2.N, tensor_data=np.array([20], dtype=self.dtype))
      expected_res = Tensor(
          data_layout=types_pb2.N, tensor_data=np.array([-50],
                                                        dtype=self.dtype))
      # res = y - 3z if x0 < x1 else y * z
      res = control_flow_ops.cond(
          math_ops.less(x0, x1), lambda: func(y, z), lambda: math_ops.mul(y, z))
    self.runAndValidate(graph, expected_res.tensor_data)

  def test_nested_cond_ops(self):
    def func_true(a, b):
      minus_one = Tensor(
          data_layout=types_pb2.N, tensor_data=np.array([-1], dtype=self.dtype))
      return control_flow_ops.cond(
          math_ops.less(a, b),
          lambda: math_ops.add(a, math_ops.mul(b, minus_one)),
          lambda: math_ops.add(a, b))

    def func_false(a, b):
      two = Tensor(
          data_layout=types_pb2.N, tensor_data=np.array([2], dtype=self.dtype))
      return control_flow_ops.cond(
          math_ops.greater(a, b), lambda: math_ops.mul(a, two),
          lambda: math_ops.mul(b, two))

    with Graph(name=self.graph_name, backend=self.backend) as graph:
      x0 = Tensor(
          data_layout=types_pb2.N, tensor_data=np.array([2], dtype=self.dtype))
      x1 = Tensor(
          data_layout=types_pb2.N, tensor_data=np.array([5], dtype=self.dtype))
      y = Tensor(
          data_layout=types_pb2.N, tensor_data=np.array([10], dtype=self.dtype))
      z = Tensor(
          data_layout=types_pb2.N, tensor_data=np.array([20], dtype=self.dtype))
      expected_res = Tensor(
          data_layout=types_pb2.N, tensor_data=np.array([40], dtype=self.dtype))
      # if x0 > x1:
      #   if y < z:
      #     res = y - z
      #   else:
      #     res = y + z
      # else:
      #   if y > z:
      #     res = 2y
      #   else:
      #     res = 2z
      res = control_flow_ops.cond(
          math_ops.greater(x0, x1), lambda: func_true(y, z),
          lambda: func_false(y, z))
    self.runAndValidate(graph, expected_res.tensor_data)

  def test_use_nested_op_result(self):
    def func_true(a, b):
      minus_one = Tensor(
          data_layout=types_pb2.N, tensor_data=np.array([-1], dtype=self.dtype))
      res = control_flow_ops.cond(
          math_ops.less(a, b),
          lambda: math_ops.add(a, math_ops.mul(b, minus_one)),
          lambda: math_ops.add(a, b))[0]
      # Use the cond results before returning.
      return math_ops.mul(res, res)

    def func_false(a, b):
      two = Tensor(
          data_layout=types_pb2.N, tensor_data=np.array([2], dtype=self.dtype))
      return control_flow_ops.cond(
          math_ops.greater(a, b), lambda: math_ops.mul(a, two),
          lambda: math_ops.mul(b, two))

    with Graph(name=self.graph_name, backend=self.backend) as graph:
      x0 = Tensor(
          data_layout=types_pb2.N, tensor_data=np.array([2], dtype=self.dtype))
      x1 = Tensor(
          data_layout=types_pb2.N, tensor_data=np.array([5], dtype=self.dtype))
      y = Tensor(
          data_layout=types_pb2.N, tensor_data=np.array([10], dtype=self.dtype))
      z = Tensor(
          data_layout=types_pb2.N, tensor_data=np.array([20], dtype=self.dtype))
      expected_res = Tensor(
          data_layout=types_pb2.N, tensor_data=np.array([100],
                                                        dtype=self.dtype))
      # if x0 < x1:
      #   if y < z:
      #     res = (y - z) ^ 2
      #   else:
      #     res = y + z
      # else:
      #   if y > z:
      #     res = 2y
      #   else:
      #     res = 2z
      res = control_flow_ops.cond(
          math_ops.less(x0, x1), lambda: func_true(y, z),
          lambda: func_false(y, z))
    self.runAndValidate(graph, expected_res.tensor_data)

if __name__ == "__main__":
  unittest.main()
