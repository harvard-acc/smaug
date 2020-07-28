#!/usr/bin/env python

""" This tests subgraphs."""

import unittest
import numpy as np

from smaug.core import types_pb2
from smaug.python.tensor import Tensor
from smaug.python.graph import Graph
from smaug.python.ops import math_ops

parent_graph_name = "parent_graph"
child_graph_name = "child_graph"
grandchild_graph_name = "grandchild_graph"
backend = "Reference"
x = Tensor(
    data_layout=types_pb2.N, tensor_data=np.random.rand(4).astype(np.float32))
y = Tensor(
    data_layout=types_pb2.N, tensor_data=np.random.rand(4).astype(np.float32))

class TestUniqueName(unittest.TestCase):
  def assertGraphContains(self, graph, node_names):
    """Test nodes in the graph."""
    self.assertEqual(
        set([
            node.name for node in graph.get_nodes() if node.op != types_pb2.Data
        ]), node_names)

  def assertNodesConnected(self, graph, child_parent_map):
    """Test the connection among nodes."""
    for child, parents in child_parent_map.items():
      child_node = graph.get_node(child)
      self.assertEqual(child_node.get_parents(), parents)

  def test_subgraph_merge(self):
    with Graph(parent_graph_name, backend) as parent_graph:
      with Graph(child_graph_name, backend) as child_graph:
        z = math_ops.add(x, y, name="add")
        w = math_ops.add(z, z, name="add_1")
    self.assertGraphContains(parent_graph, {"add", "add_1"})
    self.assertNodesConnected(parent_graph, {"add_1": ["add", "add"]})

  def test_child_use_parent_outputs(self):
    with Graph(parent_graph_name, backend) as parent_graph:
      z = math_ops.add(x, y, name="add")
      w = math_ops.add(z, z, name="add_1")
      with Graph(child_graph_name, backend) as child_graph:
        u = math_ops.mul(z, z, name="mul")
        out = math_ops.mul(w, u, name="mul_1")
    self.assertGraphContains(parent_graph, {"add", "add_1", "mul", "mul_1"})
    self.assertNodesConnected(
        parent_graph, {
            "add_1": ["add", "add"],
            "mul": ["add", "add"],
            "mul_1": ["add_1", "mul"]
        })

  def test_parent_use_child_outputs(self):
    with Graph(parent_graph_name, backend) as parent_graph:
      with Graph(child_graph_name, backend) as child_graph:
        z = math_ops.add(x, y, name="add")
        w = math_ops.add(z, z, name="add_1")
      u = math_ops.mul(z, z, name="mul")
      out = math_ops.mul(w, u, name="mul_1")
    self.assertGraphContains(parent_graph, {"add", "add_1", "mul", "mul_1"})
    self.assertNodesConnected(
        parent_graph, {
            "add_1": ["add", "add"],
            "mul": ["add", "add"],
            "mul_1": ["add_1", "mul"]
        })

  def test_nested_subgraphs(self):
    with Graph(parent_graph_name, backend) as parent_graph:
      z = math_ops.add(x, y, name="add")
      with Graph(child_graph_name, backend) as child_graph:
        w = math_ops.add(z, z, name="add_1")
        with Graph(grandchild_graph_name, backend) as grandchild_graph:
          u = math_ops.add(z, w, name="add_2")
    self.assertGraphContains(parent_graph, {"add", "add_1", "add_2"})
    self.assertNodesConnected(
        parent_graph, {
            "add_1": ["add", "add"],
            "add_2": ["add", "add_1"]
        })

if __name__ == "__main__":
  unittest.main()
