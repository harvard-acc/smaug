#!/usr/bin/env python

"""Tests for python/tensor.py."""

import unittest
import numpy as np

from smaug.python.tensor_utils import get_tensor_data
from smaug.python.graph import Graph, get_node_proto
from smaug.python.tensor import Tensor
from smaug.python.ops.data_op import input_data
from smaug.core import types_pb2

class TensorTestBase(unittest.TestCase):
  def assertEqualFP16(self, packed_fp16_data, unpacked_fp16_data):
    """Test equality between packed and unpacked float16 lists.

    Args:
      packed_fp16_data: A list of int32 values, each of which represents two
        packed float16 elements. The zeroth index fp16 value is in the lower
        16 bytes.
      unpacked_data: A numpy array of float16 values.
    """
    self.assertEqual(packed_fp16_data, list(unpacked_fp16_data.view(np.int32)))

class TensorBasicTest(TensorTestBase):
  def test_attr_reference(self):
    """Test tensor attributes with reference backend."""
    tensor_data = np.random.rand(2, 2, 4, 4).astype(np.float32)
    with Graph("test_graph", "Reference") as test_graph:
      input_tensor = Tensor(data_layout=types_pb2.NHWC, tensor_data=tensor_data)
      act = input_data(input_tensor, "input")
    graph_proto, tensor_data_array = test_graph.to_proto()
    self.assertEqual(graph_proto.backend, "Reference")
    node = get_node_proto(graph_proto, "input")
    self.assertEqual(node.input_tensors[0].data_type, types_pb2.Float32)
    self.assertEqual(node.input_tensors[0].shape.dims, [2, 2, 4, 4])
    self.assertEqual(node.input_tensors[0].shape.layout, types_pb2.NHWC)
    self.assertEqual(node.input_tensors[0].shape.alignment, 0)
    tensor_data_proto = get_tensor_data(
        tensor_data_array, node.input_tensors[0].name)
    self.assertEqual(tensor_data_proto.float_data, list(tensor_data.flatten()))
    self.assertEqual(len(tensor_data_proto.half_data), 0)
    self.assertEqual(len(tensor_data_proto.double_data), 0)
    self.assertEqual(len(tensor_data_proto.int_data), 0)
    self.assertEqual(len(tensor_data_proto.int64_data), 0)

  def test_attr_smv_no_padding(self):
    """Test tensor attributes with SMV backend. No padding is required."""
    tensor_data = np.random.rand(2, 2, 4, 8).astype(np.float16)
    with Graph("test_graph", "SMV") as test_graph:
      input_tensor = Tensor(data_layout=types_pb2.NCHW, tensor_data=tensor_data)
      act = input_data(input_tensor, "input")
    graph_proto, tensor_data_array = test_graph.to_proto()
    self.assertEqual(graph_proto.backend, "SMV")
    node = get_node_proto(graph_proto, "input")
    self.assertEqual(node.input_tensors[0].data_type, types_pb2.Float16)
    self.assertEqual(node.input_tensors[0].shape.dims, [2, 2, 4, 8])
    self.assertEqual(node.input_tensors[0].shape.layout, types_pb2.NCHW)
    self.assertEqual(node.input_tensors[0].shape.alignment, 8)
    tensor_data_proto = get_tensor_data(
        tensor_data_array, node.input_tensors[0].name)
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
      input_tensor = Tensor(data_layout=types_pb2.NCHW, tensor_data=tensor_data)
      act = input_data(input_tensor, "input")
    graph_proto, tensor_data_array = test_graph.to_proto()
    self.assertEqual(graph_proto.backend, "SMV")
    node = get_node_proto(graph_proto, "input")
    self.assertEqual(node.input_tensors[0].data_type, types_pb2.Float16)
    self.assertEqual(node.input_tensors[0].shape.dims, [2, 4])
    self.assertEqual(node.input_tensors[0].shape.layout, types_pb2.NCHW)
    self.assertEqual(node.input_tensors[0].shape.alignment, 8)
    tensor_data_proto = get_tensor_data(
        tensor_data_array, node.input_tensors[0].name)
    self.assertEqualFP16(
        tensor_data_proto.half_data,
        np.array(
            [1.1, 2.2, 3.3, 4.4, 0, 0, 0, 0, 5.5, 6.6, 7.7, 8.8, 0, 0, 0, 0],
            dtype=np.float16))
    self.assertEqual(len(tensor_data_proto.float_data), 0)
    self.assertEqual(len(tensor_data_proto.double_data), 0)
    self.assertEqual(len(tensor_data_proto.int_data), 0)
    self.assertEqual(len(tensor_data_proto.int64_data), 0)

class FP16Test(TensorTestBase):
  def test_fp16_even(self):
    """Test float16 packing when tensor's last dimension is of even size"""
    tensor_data = np.random.rand(4, 2).astype(np.float16)
    with Graph("test_graph", "Reference") as test_graph:
      input_tensor = Tensor(tensor_data=tensor_data)
      act = input_data(input_tensor, "input")
    graph_proto, tensor_data_array = test_graph.to_proto()
    node = get_node_proto(graph_proto, "input")
    self.assertEqual(node.input_tensors[0].data_type, types_pb2.Float16)
    tensor_data_proto = get_tensor_data(
        tensor_data_array, node.input_tensors[0].name)
    self.assertEqualFP16(tensor_data_proto.half_data, tensor_data.flatten())

  def test_fp16_odd(self):
    """Test float16 packing when tensor's last dimension is of odd size"""
    tensor_data = np.random.rand(4, 3).astype(np.float16)
    with Graph("test_graph", "Reference") as test_graph:
      input_tensor = Tensor(tensor_data=tensor_data)
      act = input_data(input_tensor, "input")
    graph_proto, tensor_data_array = test_graph.to_proto()
    node = get_node_proto(graph_proto, "input")
    self.assertEqual(node.input_tensors[0].data_type, types_pb2.Float16)
    tensor_data_proto = get_tensor_data(
        tensor_data_array, node.input_tensors[0].name)
    self.assertEqualFP16(tensor_data_proto.half_data, tensor_data.flatten())

  def test_fp16_odd_odd(self):
    """Test float16 packing when tensor's last dimension is of odd size.

    This tests the case when the flattened tensor is still of odd size.
    """
    tensor_data = np.random.rand(3, 3).astype(np.float16)
    with Graph("test_graph", "Reference") as test_graph:
      input_tensor = Tensor(tensor_data=tensor_data)
      act = input_data(input_tensor, "input")
    graph_proto, tensor_data_array = test_graph.to_proto()
    node = get_node_proto(graph_proto, "input")
    self.assertEqual(node.input_tensors[0].data_type, types_pb2.Float16)
    tensor_data_proto = get_tensor_data(
        tensor_data_array, node.input_tensors[0].name)
    self.assertEqualFP16(tensor_data_proto.half_data,
                         np.append(tensor_data.flatten(), np.float16(0)))

if __name__ == "__main__":
  unittest.main()
