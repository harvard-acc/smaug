"""Smaug test framework for the Python client."""

import unittest
import numpy as np

class SmaugTest(unittest.TestCase):
  """Smaug test base class.

  This class implements the common functions for different tests. Every test
  will inherit this class.
  """

  def get_node(self, graph, node_name):
    """ Find the node in the graph by its name."""
    for i in range(len(graph.nodes)):
      if graph.nodes[i].name == node_name:
        return graph.nodes[i]
    return None

  def get_tensor_data(self, tensor_data_array, tensor_name):
    """ Find the tensor data for this tensor by its name."""
    for i in range(len(tensor_data_array.data_array)):
      if tensor_data_array.data_array[i].name == tensor_name:
        return tensor_data_array.data_array[i]
    return None

  def assertEqualFP16(self, packed_fp16_data, unpacked_fp16_data):
    """Test equality between packed and unpacked float16 lists.

    Args:
      packed_fp16_data: A list of int32 values, each of which represents two
        packed float16 elements. The zeroth index fp16 value is in the lower
        16 bytes.
      unpacked_data: A numpy array of float16 values.
    """
    self.assertEqual(packed_fp16_data, list(unpacked_fp16_data.view(np.int32)))
