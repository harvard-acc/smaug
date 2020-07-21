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

def get_num_data_nodes(graph):
  count = 0
  for node in graph.get_nodes():
    if node.op == types_pb2.Data:
      count += 1
  return count

class TestUniqueName(unittest.TestCase):
  def test_auto_data_op(self):
    with Graph(graph_name, backend) as test_graph:
      x = Tensor(data_layout=types_pb2.N, tensor_data=np.array([1]))
      y = Tensor(data_layout=types_pb2.N, tensor_data=np.array([1]))
      res= math_ops.add(x, y)
    self.assertEqual(get_num_data_nodes(test_graph), 2)

  def test_no_extra_data_op(self):
    with Graph(graph_name, backend) as test_graph:
      x = Tensor(data_layout=types_pb2.N, tensor_data=np.array([1]))
      x_ = data_op.input_data(x)
      res= math_ops.add(x_, x)
    self.assertEqual(get_num_data_nodes(test_graph), 1)

  def test_use_existing_data_op(self):
    with Graph(graph_name, backend) as test_graph:
      x = Tensor(data_layout=types_pb2.N, tensor_data=np.array([1]))
      y = Tensor(data_layout=types_pb2.N, tensor_data=np.array([1]))
      x_ = data_op.input_data(x)
      res= math_ops.add(x, y)
    self.assertEqual(get_num_data_nodes(test_graph), 2)

  def test_shared_data_op(self):
    with Graph(graph_name, backend) as test_graph:
      x = Tensor(data_layout=types_pb2.N, tensor_data=np.array([1]))
      y = Tensor(data_layout=types_pb2.N, tensor_data=np.array([1]))
      res = math_ops.add(x, y)
      res = math_ops.mul(x, res)
    self.assertEqual(get_num_data_nodes(test_graph), 2)

  def test_use_existing_data_op_in_subgraph(self):
    with Graph(graph_name, backend) as parent_graph:
      x = Tensor(data_layout=types_pb2.N, tensor_data=np.array([1]))
      y = Tensor(data_layout=types_pb2.N, tensor_data=np.array([1]))
      with Graph(graph_name + "_subgraph", backend) as child_graph:
        res = math_ops.mul(x, y)
      res = math_ops.add(x, y)
    self.assertEqual(get_num_data_nodes(parent_graph), 2)

  def test_use_existing_data_op_in_parent_graph(self):
    with Graph(graph_name, backend) as parent_graph:
      x = Tensor(data_layout=types_pb2.N, tensor_data=np.array([1]))
      y = Tensor(data_layout=types_pb2.N, tensor_data=np.array([1]))
      res = math_ops.mul(x, y)
      with Graph(graph_name + "_subgraph", backend) as child_graph:
        res = math_ops.add(x, y)
    self.assertEqual(get_num_data_nodes(parent_graph), 2)

if __name__ == "__main__":
  unittest.main()
