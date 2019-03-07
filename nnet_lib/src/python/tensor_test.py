#!/usr/bin/env python

import unittest
import smaug_test
from graph import *
from tensor import *
from ops import *
from types_pb2 import *

class FP16Test(smaug_test.SmaugTest):
  def test_fp16_even(self):
    """Test float16 packing when tensor's last dimension is of even size"""
    tensor_data = np.random.rand(4, 2).astype(np.float16)
    with Graph("test_graph", "Reference") as test_graph:
      input_tensor = Tensor(
          dims=[4, 2], data_type=Float16, tensor_data=tensor_data)
      act = input_data("input", input_tensor)
    node = self.get_node(test_graph.graph, "input")
    self.assertEqual(node.input_tensors[0].data_type, Float16)
    self.assertEqual(node.input_tensors[0].half_data,
                     list(tensor_data.flatten().view(np.int32)))

  def test_fp16_odd(self):
    """Test float16 packing when tensor's last dimension is of odd size"""
    tensor_data = np.random.rand(4, 3).astype(np.float16)
    with Graph("test_graph", "Reference") as test_graph:
      input_tensor = Tensor(
          dims=[4, 3], data_type=Float16, tensor_data=tensor_data)
      act = input_data("input", input_tensor)
    node = self.get_node(test_graph.graph, "input")
    self.assertEqual(node.input_tensors[0].data_type, Float16)
    self.assertEqual(node.input_tensors[0].half_data,
                     list(tensor_data.flatten().view(np.int32)))

  def test_fp16_odd_odd(self):
    """Test float16 packing when tensor's last dimension is of odd size.

    This tests the case when the flattened tensor is still of odd size.
    """
    tensor_data = np.random.rand(3, 3).astype(np.float16)
    with Graph("test_graph", "Reference") as test_graph:
      input_tensor = Tensor(
          dims=[3, 3], data_type=Float16, tensor_data=tensor_data)
      act = input_data("input", input_tensor)
    node = self.get_node(test_graph.graph, "input")
    self.assertEqual(node.input_tensors[0].data_type, Float16)
    self.assertEqual(
        node.input_tensors[0].half_data,
        list(np.append(tensor_data.flatten(), np.float16(0)).view(np.int32)))

if __name__ == "__main__":
  unittest.main()
