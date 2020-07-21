#!/usr/bin/env python

""" This tests the unique name creation in the Graph class."""

import unittest
import numpy as np

from smaug.core import types_pb2
from smaug.python.tensor import Tensor
from smaug.python.graph import Graph
from smaug.python.ops import math_ops

graph_name = "test_graph"
backend = "Reference"
x = Tensor(
    data_layout=types_pb2.N, tensor_data=np.random.rand(4).astype(np.float32))
y = Tensor(
    data_layout=types_pb2.N, tensor_data=np.random.rand(4).astype(np.float32))

def get_node_names(graph):
  return set(
      [node.name for node in graph.get_nodes() if node.op != types_pb2.Data])

class TestUniqueName(unittest.TestCase):
  def test_default_names(self):
    with Graph(graph_name, backend) as test_graph:
      res= math_ops.add(x, y)
      res = math_ops.mul(x, res)
    self.assertEqual(get_node_names(test_graph), {"add", "mul"})

  def test_auto_unique_names(self):
    with Graph(graph_name, backend) as test_graph:
      res = math_ops.add(x, y)
      res = math_ops.add(res, res)
      res = math_ops.add(res, res)
    self.assertEqual(get_node_names(test_graph), {"add", "add_1", "add_2"})

  def test_user_supplied_names0(self):
    with Graph(graph_name, backend) as test_graph:
      res = math_ops.add(x, y, name="add")
      res = math_ops.mul(res, res, name="mul")
    self.assertEqual(get_node_names(test_graph), {"add", "mul"})

  def test_user_supplied_names1(self):
    with Graph(graph_name, backend) as test_graph:
      res = math_ops.add(x, y, name="add")
      res = math_ops.add(res, res, name="add_1")
      res = math_ops.add(res, res, name="add")
    self.assertEqual(get_node_names(test_graph), {"add", "add_1", "add_2"})

  def test_user_supplied_names1(self):
    with Graph(graph_name, backend) as test_graph:
      res = math_ops.add(x, y, name="add")
      res = math_ops.add(res, res, name="add_3")
      res = math_ops.add(res, res, name="add")
    self.assertEqual(get_node_names(test_graph), {"add", "add_1", "add_3"})

  def test_user_supplied_names2(self):
    with Graph(graph_name, backend) as test_graph:
      res = math_ops.add(x, y, name="add")
      res = math_ops.add(res, res, name="add_1")
      res = math_ops.add(res, res, name="add_1")
    self.assertEqual(get_node_names(test_graph), {"add", "add_1", "add_1_1"})

if __name__ == "__main__":
  unittest.main()
