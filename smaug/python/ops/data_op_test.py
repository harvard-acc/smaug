#!/usr/bin/env python

import unittest
import numpy as np

from smaug.core import types_pb2
from smaug.python.tensor import Tensor
from smaug.python.graph import Graph
from smaug.python.ops import data_op
from smaug.python.ops import math_ops

graph_name = "test_graph"
backend = "Reference"
x = Tensor(
    data_layout=types_pb2.N, tensor_data=np.random.rand(4).astype(np.float32))
y = Tensor(
    data_layout=types_pb2.N, tensor_data=np.random.rand(4).astype(np.float32))

def get_num_data_nodes(graph):
  return len(
      [node.name for node in graph.get_nodes() if node.op == types_pb2.Data])

class TestUniqueName(unittest.TestCase):
  def test_data_op0(self):
    with Graph(graph_name, backend) as test_graph:
      res= math_ops.add(x, y)
    self.assertEqual(get_num_data_nodes(test_graph), 2)

  def test_data_op1(self):
    with Graph(graph_name, backend) as test_graph:
      x_ = data_op.input_data(x)
      res= math_ops.add(x_, y)
    self.assertEqual(get_num_data_nodes(test_graph), 2)

  def test_data_op2(self):
    with Graph(graph_name, backend) as test_graph:
      x_ = data_op.input_data(x)
      res= math_ops.add(x, y)
    self.assertEqual(get_num_data_nodes(test_graph), 2)

  def test_data_op3(self):
    with Graph(graph_name, backend) as test_graph:
      res = math_ops.add(x, y)
      res = math_ops.mul(x, res)
    self.assertEqual(get_num_data_nodes(test_graph), 2)

if __name__ == "__main__":
  unittest.main()
