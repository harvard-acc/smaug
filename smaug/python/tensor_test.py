#!/usr/bin/env python

"""Tests for python/tensor.py."""

import unittest
import smaug_test
from graph import *
from tensor import *
from ops import *
from types_pb2 import *

class TensorBasicTest(smaug_test.SmaugTest):
  def test_attr_reference(self):
    """Test tensor attributes with reference backend."""
    tensor_data = np.random.rand(2, 2, 4, 4).astype(np.float32)
    with Graph("test_graph", "Reference") as test_graph:
      input_tensor = Tensor(data_layout=NHWC, tensor_data=tensor_data)
      act = input_data(input_tensor, "input")
    self.assertEqual(test_graph.graph.backend, "Reference")
    node = self.get_node(test_graph.graph, "input")
    self.assertEqual(node.input_tensors[0].data_type, Float32)
    self.assertEqual(node.input_tensors[0].shape.dims, [2, 2, 4, 4])
    self.assertEqual(node.input_tensors[0].shape.layout, NHWC)
    self.assertEqual(node.input_tensors[0].shape.alignment, 0)
    tensor_data_proto = self.get_tensor_data(test_graph.tensor_data_array,
                                             node.input_tensors[0].name)
    self.assertEqual(tensor_data_proto.float_data, list(tensor_data.flatten()))
    self.assertEqual(len(tensor_data_proto.half_data), 0)
    self.assertEqual(len(tensor_data_proto.double_data), 0)
    self.assertEqual(len(tensor_data_proto.int_data), 0)
    self.assertEqual(len(tensor_data_proto.int64_data), 0)

  def test_attr_smv_no_padding(self):
    """Test tensor attributes with SMV backend. No padding is required."""
    tensor_data = np.random.rand(2, 2, 4, 8).astype(np.float16)
    with Graph("test_graph", "SMV") as test_graph:
      input_tensor = Tensor(data_layout=NCHW, tensor_data=tensor_data)
      act = input_data(input_tensor, "input")
    self.assertEqual(test_graph.graph.backend, "SMV")
    node = self.get_node(test_graph.graph, "input")
    self.assertEqual(node.input_tensors[0].data_type, Float16)
    self.assertEqual(node.input_tensors[0].shape.dims, [2, 2, 4, 8])
    self.assertEqual(node.input_tensors[0].shape.layout, NCHW)
    self.assertEqual(node.input_tensors[0].shape.alignment, 8)
    tensor_data_proto = self.get_tensor_data(test_graph.tensor_data_array,
                                             node.input_tensors[0].name)
    self.assertEqualFP16(tensor_data_proto.half_data, tensor_data.flatten())
    self.assertEqual(len(tensor_data_proto.float_data), 0)
    self.assertEqual(len(tensor_data_proto.double_data), 0)
    self.assertEqual(len(tensor_data_proto.int_data), 0)
    self.assertEqual(len(tensor_data_proto.int64_data), 0)

  def test_attr_smv_padding(self):
    """Test tensor attributes with SMV backend. Additional padding required."""
    tensor_data = np.array([[1.1, 2.2, 3.3, 4.4], [5.5, 6.6, 7.7, 8.8]],
                           dtype=np.float16)
    with Graph("test_graph", "SMV") as test_graph:
      input_tensor = Tensor(data_layout=NCHW, tensor_data=tensor_data)
      act = input_data(input_tensor, "input")
    self.assertEqual(test_graph.graph.backend, "SMV")
    node = self.get_node(test_graph.graph, "input")
    self.assertEqual(node.input_tensors[0].data_type, Float16)
    self.assertEqual(node.input_tensors[0].shape.dims, [2, 4])
    self.assertEqual(node.input_tensors[0].shape.layout, NCHW)
    self.assertEqual(node.input_tensors[0].shape.alignment, 8)
    tensor_data_proto = self.get_tensor_data(test_graph.tensor_data_array,
                                             node.input_tensors[0].name)
    self.assertEqualFP16(
        tensor_data_proto.half_data,
        np.array(
            [1.1, 2.2, 3.3, 4.4, 0, 0, 0, 0, 5.5, 6.6, 7.7, 8.8, 0, 0, 0, 0],
            dtype=np.float16))
    self.assertEqual(len(tensor_data_proto.float_data), 0)
    self.assertEqual(len(tensor_data_proto.double_data), 0)
    self.assertEqual(len(tensor_data_proto.int_data), 0)
    self.assertEqual(len(tensor_data_proto.int64_data), 0)

class FP16Test(smaug_test.SmaugTest):
  def test_fp16_even(self):
    """Test float16 packing when tensor's last dimension is of even size"""
    tensor_data = np.random.rand(4, 2).astype(np.float16)
    with Graph("test_graph", "Reference") as test_graph:
      input_tensor = Tensor(tensor_data=tensor_data)
      act = input_data(input_tensor, "input")
    node = self.get_node(test_graph.graph, "input")
    self.assertEqual(node.input_tensors[0].data_type, Float16)
    tensor_data_proto = self.get_tensor_data(test_graph.tensor_data_array,
                                             node.input_tensors[0].name)
    self.assertEqualFP16(tensor_data_proto.half_data, tensor_data.flatten())

  def test_fp16_odd(self):
    """Test float16 packing when tensor's last dimension is of odd size"""
    tensor_data = np.random.rand(4, 3).astype(np.float16)
    with Graph("test_graph", "Reference") as test_graph:
      input_tensor = Tensor(tensor_data=tensor_data)
      act = input_data(input_tensor, "input")
    node = self.get_node(test_graph.graph, "input")
    self.assertEqual(node.input_tensors[0].data_type, Float16)
    tensor_data_proto = self.get_tensor_data(test_graph.tensor_data_array,
                                             node.input_tensors[0].name)
    self.assertEqualFP16(tensor_data_proto.half_data, tensor_data.flatten())

  def test_fp16_odd_odd(self):
    """Test float16 packing when tensor's last dimension is of odd size.

    This tests the case when the flattened tensor is still of odd size.
    """
    tensor_data = np.random.rand(3, 3).astype(np.float16)
    with Graph("test_graph", "Reference") as test_graph:
      input_tensor = Tensor(tensor_data=tensor_data)
      act = input_data(input_tensor, "input")
    node = self.get_node(test_graph.graph, "input")
    self.assertEqual(node.input_tensors[0].data_type, Float16)
    tensor_data_proto = self.get_tensor_data(test_graph.tensor_data_array,
                                             node.input_tensors[0].name)
    self.assertEqualFP16(tensor_data_proto.half_data,
                         np.append(tensor_data.flatten(), np.float16(0)))

if __name__ == "__main__":
  unittest.main()
